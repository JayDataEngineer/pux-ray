# =============================================================================
# Wan2GP Studio Web — Next.js + assistant-ui + Shadcn
#
# Two-stage build: Node.js builder → slim runtime
# =============================================================================

# Builder stage
FROM node:22-slim AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

RUN npm run build

# Final stage
FROM node:22-slim

WORKDIR /app

COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000

ENV HOSTNAME=0.0.0.0
ENV PORT=3000

CMD ["node", "server.js"]
