import fs from "node:fs";
import path from "node:path";

function arg(name, fallback = null) {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx === -1) return fallback;
  return process.argv[idx + 1] ?? fallback;
}

const runArg = arg("run", "latest");
const iteration = arg("iteration", null);

let runId = runArg;
if (runArg === "latest") {
  runId = fs.readFileSync(".agent-runs/latest", "utf8").trim();
}

const runDir = path.join(".agent-runs", runId);
const targets = iteration
  ? [path.join(runDir, `iteration-${String(iteration).padStart(2, "0")}`)]
  : fs.readdirSync(runDir)
      .filter((entry) => /^iteration-\d+$/.test(entry))
      .map((entry) => path.join(runDir, entry));

const artifactSchemas = new Map([
  ["01-plan.json", ".agent-loop/schemas/plan.schema.json"],
  ["02-test-plan.json", ".agent-loop/schemas/test-plan.schema.json"],
  ["03-implementation-report.json", ".agent-loop/schemas/implementation-report.schema.json"],
  ["04-audit-report.json", ".agent-loop/schemas/audit-report.schema.json"]
]);

function loadJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function validate(value, schema, location) {
  const errors = [];
  const type = schema.type;

  if (type === "object" && (value === null || Array.isArray(value) || typeof value !== "object")) {
    return [`${location}: expected object`];
  }
  if (type === "array" && !Array.isArray(value)) {
    return [`${location}: expected array`];
  }
  if (type === "string" && typeof value !== "string") {
    return [`${location}: expected string`];
  }

  if (schema.enum && !schema.enum.includes(value)) {
    errors.push(`${location}: expected one of ${schema.enum.join(", ")}`);
  }

  if (schema.required && typeof value === "object" && value !== null && !Array.isArray(value)) {
    for (const key of schema.required) {
      if (!(key in value)) {
        errors.push(`${location}.${key}: missing required field`);
      }
    }
  }

  if (schema.properties && typeof value === "object" && value !== null && !Array.isArray(value)) {
    for (const [key, childSchema] of Object.entries(schema.properties)) {
      if (key in value) {
        errors.push(...validate(value[key], childSchema, `${location}.${key}`));
      }
    }
  }

  if (schema.items && Array.isArray(value)) {
    value.forEach((item, index) => {
      errors.push(...validate(item, schema.items, `${location}[${index}]`));
    });
  }

  return errors;
}

let errorCount = 0;
for (const dir of targets) {
  if (!fs.existsSync(dir)) continue;
  for (const [artifact, schemaPath] of artifactSchemas) {
    const artifactPath = path.join(dir, artifact);
    if (!fs.existsSync(artifactPath)) continue;
    const errors = validate(loadJson(artifactPath), loadJson(schemaPath), artifactPath);
    if (errors.length) {
      errorCount += errors.length;
      console.error(errors.join("\n"));
    } else {
      console.log(`${artifactPath}: ok`);
    }
  }
}

if (errorCount) {
  process.exit(1);
}
