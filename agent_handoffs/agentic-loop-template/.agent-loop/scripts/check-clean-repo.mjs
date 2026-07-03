import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

function readConfig() {
  const configPath = ".agent-loop/config/loop.config.json";
  if (!fs.existsSync(configPath)) return {};
  return JSON.parse(fs.readFileSync(configPath, "utf8"));
}

function latestRunDir() {
  const latestPath = ".agent-runs/latest";
  if (!fs.existsSync(latestPath)) return ".agent-runs";
  const runId = fs.readFileSync(latestPath, "utf8").trim();
  return runId ? path.join(".agent-runs", runId) : ".agent-runs";
}

try {
  const config = readConfig();
  const status = execSync("git status --short", { encoding: "utf8" }).trim();
  if (status) {
    if (config.require_clean_git_status_before_iteration !== false) {
      console.error("Repo is not clean:");
      console.error(status);
      process.exit(1);
    }

    const runDir = latestRunDir();
    fs.mkdirSync(runDir, { recursive: true });
    const snapshotPath = path.join(runDir, "dirty-state-snapshot.txt");
    fs.writeFileSync(snapshotPath, status + "\n", "utf8");
    console.log(`Repo is dirty; wrote ${snapshotPath}.`);
    process.exit(0);
  }
  console.log("Repo is clean.");
} catch (err) {
  console.error("Could not check git status. Is this a git repo?");
  process.exit(1);
}
