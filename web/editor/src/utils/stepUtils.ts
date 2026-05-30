import type { InputSpec, StepSpecInfo } from '../types'

const INPUT_RE = /\{\{\s*inputs\.(\w+)\s*\}\}/g
const SOURCE_RE = /\{\{\s*(\w+)\.outputs\.(\w+)\s*\}\}/g

function collectMatches(val: unknown, re: RegExp, out: Set<string>) {
  if (typeof val === 'string') {
    const r = new RegExp(re.source, re.flags)
    let m
    while ((m = r.exec(val)) !== null) out.add(m[1])
  } else if (Array.isArray(val)) {
    for (const item of val) collectMatches(item, re, out)
  }
}

/** Extract {{ inputs.X }} refs from a step's params AND model field */
export function getInputsForStep(step: StepSpecInfo): Set<string> {
  const inputs = new Set<string>()
  if (step.params) {
    for (const val of Object.values(step.params)) {
      collectMatches(val, INPUT_RE, inputs)
    }
  }
  if (step.model) collectMatches(step.model, INPUT_RE, inputs)
  return inputs
}

/** Extract {{ stepId.outputs.key }} refs from a step's params */
export function getSourceRefs(step: StepSpecInfo): Array<{ stepId: string; outputKey: string }> {
  const refs: Array<{ stepId: string; outputKey: string }> = []
  if (!step.params) return refs
  for (const val of Object.values(step.params)) {
    if (typeof val === 'string') {
      const r = new RegExp(SOURCE_RE.source, SOURCE_RE.flags)
      let m
      while ((m = r.exec(val)) !== null) {
        if (m[1] !== 'inputs') refs.push({ stepId: m[1], outputKey: m[2] })
      }
    } else if (Array.isArray(val)) {
      for (const item of val) {
        if (typeof item === 'string') {
          const r = new RegExp(SOURCE_RE.source, SOURCE_RE.flags)
          let m
          while ((m = r.exec(item)) !== null) {
            if (m[1] !== 'inputs') refs.push({ stepId: m[1], outputKey: m[2] })
          }
        }
      }
    }
  }
  return refs
}

/** Map each spec input name to the step IDs that reference it */
export function buildInputStepMap(
  specInputs: Record<string, InputSpec>,
  steps: StepSpecInfo[],
): Map<string, string[]> {
  const map = new Map<string, string[]>()
  for (const name of Object.keys(specInputs)) {
    const users: string[] = []
    for (const step of steps) {
      if (getInputsForStep(step).has(name)) users.push(step.id)
    }
    map.set(name, users)
  }
  return map
}
