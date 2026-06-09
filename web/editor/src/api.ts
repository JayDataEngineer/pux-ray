// Re-export from MCP client — all workflow API calls now go through MCP tools
// Plus REST API for service catalog and direct invocation
export {
  // Workflow
  listSpecs,
  getSpec,
  startRun,
  getRun,
  cancelRun,
  rerunStep,
  executeStep,
  approveStep,
  continueStep,
  artifactUrl,
  sseUrl,
  loadKimodo,
  kimodoUrl,
  // Service catalog
  listServices,
  getServiceInfo,
  invokeService,
  invokeServiceFormData,
  loadService,
  forgeStatus,
  fileToBase64,
} from './mcp'
