import { callMcpTool } from "@/lib/mcp-client";

export async function POST(req: Request) {
  try {
    const { image_b64, steps, guidance, seed, resolution } = await req.json();

    if (!image_b64) {
      return Response.json(
        { error: "Missing 'image_b64' parameter" },
        { status: 400 },
      );
    }

    const params: Record<string, unknown> = {
      model: "trellis",
      image_b64,
    };
    if (steps) params.sampling_steps = steps;
    if (guidance) params.guide_scale = guidance;
    if (seed != null) params.seed = seed;
    if (resolution) params.resolution = resolution;

    const result = await callMcpTool("run", {
      service: "wan2gp",
      params,
    });
    return Response.json(result);
  } catch (e: any) {
    return Response.json(
      { error: e.message || "3D generation failed" },
      { status: 500 },
    );
  }
}
