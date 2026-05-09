from __future__ import annotations

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from registry.config import Config

logger = logging.getLogger(__name__)

_AGTYPE_RE = re.compile(r"^(.*)::(vertex|edge|path)\s*$")

_AGE_QUERY = """
SELECT *
FROM ag_catalog.cypher('{graph_name}', $AGE${statement}$AGE$)
{column_def}
"""

_AGE_GRAPH_QUERY = """
SELECT * FROM ag_catalog.create_graph('{graph_name}');
"""

_AGE_LABEL_QUERY = """
SELECT *
FROM ag_catalog.cypher('{graph_name}', $AGE$MATCH (n) RETURN labels(n) AS label, count(*) AS cnt$AGE$)
AS (label agtype, cnt agtype);
"""


def _parse_agtype(raw: Any) -> Any:
    if raw is None:
        return None
    text = str(raw).strip()
    m = _AGTYPE_RE.match(text)
    if m:
        text = m.group(1).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _dict_from_agtype_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: _parse_agtype(v) for k, v in row.items()}


class GraphDB:
    """PostgreSQL + Apache AGE graph database connector.

    Provides connection pooling, Cypher graph query execution,
    and AGE result parsing. Use as a context manager or global
    singleton via :func:`graph_db`.

    Usage::

        async with graph_db() as db:
            result = await db.query("MATCH (n) RETURN n LIMIT 5")
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        user: str | None = None,
        password: str | None = None,
        dbname: str | None = None,
        graph_name: str = "tech_noir_graph",
        min_conn: int = 2,
        max_conn: int = 10,
        autosetup: bool = True,
    ) -> None:
        cfg = Config()
        self._host = host or os.environ.get("PGHOST") or cfg.get("database.host", "postgres.infra.svc.cluster.local")
        self._port = port or int(os.environ.get("PGPORT") or cfg.get("database.port", 5432))
        self._user = user or os.environ.get("PGUSER") or cfg.get("database.user", "postgres")
        self._password = password or os.environ.get("PGPASSWORD") or os.environ.get("POSTGRES_PASSWORD") or cfg.get("database.password", "")
        self._dbname = dbname or os.environ.get("PGDATABASE") or cfg.get("database.dbname", "tech_noir")
        self._graph_name = graph_name
        self._min_conn = min_conn or int(os.environ.get("PG_MIN_CONN") or cfg.get("database.min_conn", 2))
        self._max_conn = max_conn or int(os.environ.get("PG_MAX_CONN") or cfg.get("database.max_conn", 10))
        self._pool: AsyncConnectionPool | None = None
        self._autosetup = autosetup

    @property
    def dsn(self) -> str:
        pw = f":{self._password}" if self._password else ""
        return f"postgres://{self._user}{pw}@{self._host}:{self._port}/{self._dbname}"

    async def _ensure_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            self._pool = AsyncConnectionPool(
                self.dsn,
                min_size=self._min_conn,
                max_size=self._max_conn,
                open=False,
            )
            await self._pool.open()
            logger.info("Connected: %s@%s:%s/%s graph=%s",
                        self._user, self._host, self._port, self._dbname, self._graph_name)
            if self._autosetup:
                await self._setup()
        return self._pool

    async def _setup(self) -> None:
        async with self._conn() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS age;")
            row = await conn.execute("SELECT graphid FROM ag_catalog.ag_graph WHERE name = %s;", (self._graph_name,))
            exists = await row.fetchone()
            if not exists:
                stmt = _AGE_GRAPH_QUERY.format(graph_name=self._graph_name)
                await conn.execute(stmt)
                logger.info("Created graph: %s", self._graph_name)
            await conn.execute("LOAD 'age';")

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[AsyncConnection]:
        pool = await self._ensure_pool()
        async with pool.connection() as conn:
            await conn.set_autocommit(True)
            await conn.execute("SET search_path TO ag_catalog, public;")
            async with conn.cursor(row_factory=dict_row) as cur:
                yield cur

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Disconnected from %s", self._host)

    async def execute(self, sql: str, params: tuple | None = None, *, fetch: bool = False) -> list[dict[str, Any]]:
        async with self._conn() as cur:
            await cur.execute(sql, params)
            if fetch:
                rows = await cur.fetchall()
                return [_dict_from_agtype_row(r) for r in rows]
            return []

    async def query(self, statement: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        cypher = statement.strip()
        col_def = _infer_columns(cypher) or "(v agtype)"
        stmt = _AGE_QUERY.format(graph_name=self._graph_name, statement=cypher, column_def=f"AS {col_def}")
        return await self.execute(stmt, fetch=True)

    async def schema(self) -> dict[str, Any]:
        stmt = _AGE_LABEL_QUERY.format(graph_name=self._graph_name)
        labels = await self.execute(stmt, fetch=True)
        vertices: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for row in labels:
            label_list = _parse_agtype(row.get("label")) or []
            label = label_list[0] if isinstance(label_list, list) and label_list else str(row.get("label", ""))
            cnt = _parse_agtype(row.get("cnt")) or 0
            entry = {"label": str(label).replace('"', ""), "count": int(cnt) if isinstance(cnt, (int, float)) else 0}
            vertices.append(entry)
        return {"graph": self._graph_name, "vertices": vertices, "edges": edges}


def _infer_columns(cypher: str) -> str | None:
    m = re.search(r"RETURN\s+(.+?)(?:\s+ORDER\s+BY|\s+LIMIT|\s+SKIP|\s*$)", cypher, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    expr_str = m.group(1).strip()
    parts = _split_cypher_return(expr_str)
    cols = []
    for i, part in enumerate(parts):
        part = part.strip()
        alias = re.search(r"\s+AS\s+(\w+)\s*$", part, re.IGNORECASE)
        if alias:
            name = alias.group(1)
        else:
            name = f"col{i}"
        cols.append(f"{name} agtype")
    if cols:
        return "(" + ", ".join(cols) + ")"
    return None


def _split_cypher_return(expr_str: str) -> list[str]:
    """Split RETURN expressions on commas, respecting parens for function calls."""
    parts = []
    depth = 0
    current: list[str] = []
    for ch in expr_str:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    remaining = "".join(current).strip()
    if remaining:
        parts.append(remaining)
    return parts


_graph_db_instance: GraphDB | None = None


async def graph_db(**kwargs: Any) -> GraphDB:
    global _graph_db_instance
    if _graph_db_instance is None:
        _graph_db_instance = GraphDB(**kwargs)
    return _graph_db_instance
