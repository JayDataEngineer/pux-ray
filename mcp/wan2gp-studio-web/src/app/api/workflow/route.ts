import { callMcpTool } from "@/lib/mcp-client";

// Unified workflow API — dispatches to the correct MCP tool by action
export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { action } = body;

    switch (action) {
      case "list_specs":
        return Response.json(await callMcpTool("workflow_list_specs", {}));

      case "get_spec":
        return Response.json(await callMcpTool("workflow_get_spec", { spec_name: body.spec_name }));

      case "start_run":
        return Response.json(await callMcpTool("workflow_start_run", {
          spec_name: body.spec_name,
          inputs: body.inputs || {},
          manual: body.manual ?? true,
        }));

      case "get_run":
        return Response.json(await callMcpTool("workflow_get_run", {
          spec_name: body.spec_name,
          run_id: body.run_id,
        }));

      case "execute_step":
        return Response.json(await callMcpTool("workflow_execute_step", {
          spec_name: body.spec_name,
          run_id: body.run_id,
          step_id: body.step_id,
          params: body.params || {},
        }));

      case "approve_step":
        return Response.json(await callMcpTool("workflow_approve_step", {
          spec_name: body.spec_name,
          run_id: body.run_id,
          step_id: body.step_id,
          data: body.data || {},
        }));

      case "rerun_step":
        return Response.json(await callMcpTool("workflow_rerun_step", {
          spec_name: body.spec_name,
          run_id: body.run_id,
          step_id: body.step_id,
          params: body.params || {},
        }));

      case "cancel_run":
        return Response.json(await callMcpTool("workflow_cancel_run", {
          spec_name: body.spec_name,
          run_id: body.run_id,
        }));

      default:
        return Response.json({ error: `Unknown action: ${action}` }, { status: 400 });
    }
  } catch (e: unknown) {
    return Response.json(
      { error: e instanceof Error ? e.message : "Workflow action failed" },
      { status: 500 },
    );
  }
}
