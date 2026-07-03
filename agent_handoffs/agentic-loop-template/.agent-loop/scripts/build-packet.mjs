import fs from "node:fs";
import path from "node:path";
import { execSync } from "node:child_process";

function arg(name, fallback = null) {
  const idx = process.argv.indexOf(`--${name}`);
  if (idx === -1) return fallback;
  return process.argv[idx + 1] ?? fallback;
}

const runArg = arg("run", "latest");
const agent = arg("agent");
const iteration = arg("iteration", null);

if (!agent) {
  console.error("Missing --agent. Use planner, test-designer, coder, auditor, or governor.");
  process.exit(1);
}

let runId = runArg;
if (runArg === "latest") {
  runId = fs.readFileSync(".agent-runs/latest", "utf8").trim();
}

const agentMap = {
  "planner": ".agent-loop/agents/01-planner.md",
  "test-designer": ".agent-loop/agents/02-test-designer.md",
  "coder": ".agent-loop/agents/03-coder.md",
  "auditor": ".agent-loop/agents/04-adversarial-auditor.md",
  "governor": ".agent-loop/agents/05-governor.md"
};

const promptPath = agentMap[agent];
if (!promptPath || !fs.existsSync(promptPath)) {
  console.error(`Unknown agent: ${agent}`);
  process.exit(1);
}

const runDir = path.join(".agent-runs", runId);
const packetDir = iteration
  ? path.join(runDir, `iteration-${String(iteration).padStart(2, "0")}`)
  : runDir;

fs.mkdirSync(packetDir, { recursive: true });

function readIfExists(file) {
  return fs.existsSync(file) ? fs.readFileSync(file, "utf8") : "";
}

function findProjectRoot(startDir) {
  let current = path.resolve(startDir);
  while (true) {
    const hasExamBankProject =
      fs.existsSync(path.join(current, "pyproject.toml")) &&
      fs.existsSync(path.join(current, "src", "exam_bank"));
    if (hasExamBankProject) return current;
    const parent = path.dirname(current);
    if (parent === current) return path.resolve(startDir);
    current = parent;
  }
}

function readProjectIfExists(root, file) {
  const resolved = path.join(root, file);
  return fs.existsSync(resolved) ? fs.readFileSync(resolved, "utf8") : "";
}

function gitStatus(root) {
  try {
    return execSync("git status --short", { cwd: root, encoding: "utf8" }).trim();
  } catch {
    return "";
  }
}

const sharedFiles = [
  ".agent-loop/OBJECTIVE.md",
  ".agent-loop/BACKLOG.md",
  ".agent-loop/config/planner-purpose.md",
  ".agent-loop/policies/definition-of-done.md",
  ".agent-loop/policies/repo-hygiene.md",
  ".agent-loop/policies/protected-files.md",
  ".agent-loop/config/loop.config.json"
];

const projectRoot = findProjectRoot(process.cwd());
const projectFiles = [
  "README.md",
  "ARCHITECTURE.md",
  "ROADMAP.md",
  "docs/COMMAND_ATLAS.md",
  "docs/AUTO_TRIAGE.md",
  "docs/TRIAGE_WORKFLOW.md",
  "docs/TOPIC_ROUTING_SIDECAR_CONTRACT.md"
];

let packet = "";
packet += `# Agent Packet\n\n`;
packet += `Run ID: ${runId}\n`;
packet += iteration ? `Iteration: ${iteration}\n` : "";
packet += `Agent: ${agent}\n\n`;

packet += `## Agent Instructions\n\n${readIfExists(promptPath)}\n\n`;

packet += `## Shared Project Loop Context\n\n`;
for (const file of sharedFiles) {
  packet += `### ${file}\n\n`;
  packet += readIfExists(file) || "(missing)\n";
  packet += "\n";
}

packet += `## Current Project Context\n\n`;
packet += `Project root: ${projectRoot}\n\n`;
packet += `### git status --short\n\n`;
packet += gitStatus(projectRoot) || "(clean or unavailable)";
packet += "\n\n";
for (const file of projectFiles) {
  packet += `### ${file}\n\n`;
  packet += readProjectIfExists(projectRoot, file) || "(missing)\n";
  packet += "\n";
}

packet += `## Prior Artifacts\n\n`;
if (iteration) {
  const iterNum = Number(iteration);
  for (let i = 1; i <= iterNum; i += 1) {
    const iter = String(i).padStart(2, "0");
    const dir = path.join(runDir, `iteration-${iter}`);
    if (!fs.existsSync(dir)) continue;
    packet += `### iteration-${iter}\n\n`;
    for (const file of fs.readdirSync(dir).sort()) {
      if (file.endsWith(".json") || file.endsWith(".md") || file.endsWith(".txt")) {
        packet += `#### ${file}\n\n`;
        packet += readIfExists(path.join(dir, file));
        packet += "\n\n";
      }
    }
  }
} else {
  for (const file of fs.readdirSync(runDir).sort()) {
    if (file.endsWith(".json") || file.endsWith(".md") || file.endsWith(".txt")) {
      packet += `#### ${file}\n\n`;
      packet += readIfExists(path.join(runDir, file));
      packet += "\n\n";
    }
  }
}

packet += `## Required Behavior\n\n`;
packet += `Inspect the actual repo before acting. Follow the agent instructions exactly. Write the required artifact into the run folder. Keep source changes bounded and repo hygiene high.\n`;

const outFile = path.join(packetDir, `${agent}-packet.md`);
fs.writeFileSync(outFile, packet);
console.log(outFile);
