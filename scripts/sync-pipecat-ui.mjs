#!/usr/bin/env node
// Refreshes the vendored Pipecat UI snapshot the CLI copies into generated React
// clients (src/pipecat/cli/templates/client/_pipecat_ui/).
//
// Pipecat UI (https://ui.pipecat.ai) is a shadcn registry: components install as
// source through the shadcn CLI rather than as an npm package. `pipecat init` must
// produce a project that builds with no network step beyond `npm install`, so the
// templates carry a snapshot of that installed source. This script rebuilds the
// snapshot by running the real install into a throwaway shadcn project and
// copying the result back, so the vendored files are exactly what the shadcn CLI
// produces.
//
// The registry at ui.pipecat.ai serves pipecat-ui's current main with no
// per-release payloads, so by default the snapshot is built from a pipecat-ui
// release tag instead: the repository is cloned at the tag, its registry JSON is
// built with the shadcn CLI and served locally for the install. dependencies.json
// records the ref and commit the snapshot came from.
//
// Usage (from the repository root, Node 22+, npm and git required):
//
//   node scripts/sync-pipecat-ui.mjs                    # latest pipecat-ui release
//   node scripts/sync-pipecat-ui.mjs --tag v1.0.0       # a specific release
//   node scripts/sync-pipecat-ui.mjs --ref my-branch    # an unreleased branch or tag
//   node scripts/sync-pipecat-ui.mjs --repo ../pipecat-ui --ref my-branch
//                                                       # ... from a local clone
//   node scripts/sync-pipecat-ui.mjs --live             # ui.pipecat.ai as it is today
//   node scripts/sync-pipecat-ui.mjs --keep             # leave the scratch project behind
//
// Review the resulting diff, then run the client generation tests and build a
// generated project before committing.

import { spawn } from "node:child_process";
import { once } from "node:events";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";

const PIPECAT_UI_REPOSITORY = "https://github.com/pipecat-ai/pipecat-ui.git";
const LIVE_REGISTRY_URL = "https://ui.pipecat.ai/r/{name}.json";

// The shadcn CLI is pinned because its presets decide the shape of the installed
// primitives and helpers (for example, the nova preset writes `lib/utils.ts` as a
// re-export of the `cn` package). Bump it deliberately and review the diff.
const SHADCN_VERSION = "4.21.0";

// Registry items the generated client composes: the console block (which pulls
// in every component it is built from) and the shadcn select the template's own
// transport picker uses.
const ITEMS = ["@pipecat/console", "select"];

// Packages the templates declare themselves; everything else the install pulls in
// is recorded in dependencies.json for package.json.jinja2 to render.
const TEMPLATE_OWNED = new Set(["react", "react-dom"]);

// Installed by the shadcn preset; the template stylesheets import the fonts
// they use directly and the package templates list them.
const EXCLUDED = new Set(["@fontsource-variable/geist"]);

const { values: options } = parseArgs({
  options: {
    tag: { type: "string" },
    ref: { type: "string" },
    repo: { type: "string", default: PIPECAT_UI_REPOSITORY },
    live: { type: "boolean", default: false },
    keep: { type: "boolean", default: false },
  },
});
if ([options.live, options.tag, options.ref].filter(Boolean).length > 1) {
  throw new Error("--live, --tag and --ref are mutually exclusive");
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.join(repoRoot, "src/pipecat/cli/templates/client/_pipecat_ui");

async function run(command, args, cwd, { capture = false } = {}) {
  const child = spawn(command, args, {
    cwd,
    stdio: capture ? ["ignore", "pipe", "inherit"] : "inherit",
    env: {
      ...process.env,
      CI: "true",
      npm_config_audit: "false",
      npm_config_fund: "false",
    },
  });
  let stdout = "";
  child.stdout?.on("data", (chunk) => (stdout += chunk));
  const [code] = await once(child, "exit");
  if (code !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed in ${cwd}`);
  }
  return stdout;
}

/** Highest semver tag of the form vX.Y.Z on the pipecat-ui remote. */
async function latestTag() {
  const refs = await run("git", ["ls-remote", "--tags", "--refs", PIPECAT_UI_REPOSITORY], repoRoot, {
    capture: true,
  });
  const tags = refs
    .split("\n")
    .map((line) => line.split("refs/tags/")[1])
    .filter((tag) => /^v\d+\.\d+\.\d+$/.test(tag ?? ""))
    .sort((a, b) => {
      const [x, y] = [a, b].map((t) => t.slice(1).split(".").map(Number));
      return x[0] - y[0] || x[1] - y[1] || x[2] - y[2];
    });
  if (!tags.length) throw new Error("No release tags found on pipecat-ui");
  return tags.at(-1);
}

/** Serves `<dir>/<name>.json` as a shadcn registry on a loopback port. */
async function serveRegistry(dir) {
  const server = http.createServer((req, res) => {
    const name = /^\/r\/([a-z0-9-]+)\.json$/.exec(req.url ?? "")?.[1];
    const file = name && path.join(dir, `${name}.json`);
    if (!file || !fs.existsSync(file)) {
      res.writeHead(404).end();
      return;
    }
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(fs.readFileSync(file));
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  return { server, url: `http://127.0.0.1:${server.address().port}/r/{name}.json` };
}

const scratch = fs.mkdtempSync(path.join(os.tmpdir(), "pipecat-ui-sync-"));
console.log(`Scratch project: ${scratch}`);

const write = (file, content) => {
  const target = path.join(scratch, file);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(
    target,
    typeof content === "string" ? content : JSON.stringify(content, null, 2) + "\n",
  );
};

let registryServer;
let succeeded = false;
try {
  // Minimal Vite + Tailwind v4 consumer, the shape the shadcn CLI expects to find.
  write("package.json", {
    name: "pipecat-ui-sync",
    private: true,
    type: "module",
    dependencies: {
      react: "latest",
      "react-dom": "latest",
      clsx: "latest",
      "tailwind-merge": "latest",
      "tw-animate-css": "latest",
      tailwindcss: "latest",
    },
    devDependencies: {
      "@tailwindcss/vite": "latest",
      "@types/react": "latest",
      "@types/react-dom": "latest",
      "@vitejs/plugin-react": "latest",
      shadcn: SHADCN_VERSION,
      typescript: "latest",
      vite: "latest",
    },
  });
  write("tsconfig.json", {
    compilerOptions: {
      target: "ES2023",
      lib: ["ES2023", "DOM"],
      module: "ESNext",
      moduleResolution: "Bundler",
      jsx: "react-jsx",
      types: ["vite/client"],
      strict: true,
      skipLibCheck: true,
      noEmit: true,
      paths: { "@/*": ["./src/*"] },
    },
    include: ["src"],
  });
  write(
    "vite.config.mjs",
    'import { defineConfig } from "vite";\nimport react from "@vitejs/plugin-react";\nimport tailwind from "@tailwindcss/vite";\nexport default defineConfig({ plugins: [react(), tailwind()], resolve: { alias: { "@": new URL("./src", import.meta.url).pathname } } });\n',
  );
  write("src/index.css", '@import "tailwindcss";\n');
  write("src/main.tsx", "export {};\n");
  write(
    "index.html",
    '<html><body><div id="root"></div><script type="module" src="/src/main.tsx"></script></body></html>\n',
  );

  await run("npm", ["install", "--ignore-scripts", "--no-audit", "--no-fund"], scratch);
  const cli = path.join(scratch, "node_modules/shadcn/dist/index.js");

  // Where the @pipecat items come from: a registry built from a release tag,
  // served locally, or the live site.
  let source;
  let registryUrl;
  if (options.live) {
    source = { registry: LIVE_REGISTRY_URL, syncedAt: new Date().toISOString() };
    registryUrl = LIVE_REGISTRY_URL;
  } else {
    const ref = options.ref ?? options.tag ?? (await latestTag());
    const repository = path.isAbsolute(options.repo) || options.repo.startsWith(".")
      ? path.resolve(options.repo)
      : options.repo;
    const clone = path.join(scratch, "pipecat-ui");
    await run("git", ["clone", "--quiet", "--depth", "1", "--branch", ref, repository, clone], scratch);
    const commit = (await run("git", ["rev-parse", "HEAD"], clone, { capture: true })).trim();
    const built = path.join(scratch, "registry");
    await run(process.execPath, [cli, "build", "--output", built], path.join(clone, "packages/registry"));
    registryServer = await serveRegistry(built);
    registryUrl = registryServer.url;
    source = { repository, ref, commit };
    console.log(`Registry built from pipecat-ui ${ref} (${commit})`);
  }

  await run(
    process.execPath,
    [
      cli,
      "init",
      "--preset",
      "nova",
      "--base",
      "base",
      "--template",
      "vite",
      "--yes",
      "--force",
      "--no-monorepo",
      "--no-reinstall",
      "--no-rtl",
      "--no-pointer",
    ],
    scratch,
  );

  const componentsJsonPath = path.join(scratch, "components.json");
  const componentsJson = JSON.parse(fs.readFileSync(componentsJsonPath, "utf8"));
  if (componentsJson.style !== "base-nova") {
    throw new Error(`Expected base-nova style, got ${componentsJson.style}`);
  }
  componentsJson.registries = { ...componentsJson.registries, "@pipecat": registryUrl };
  write("components.json", componentsJson);

  await run(process.execPath, [cli, "add", ...ITEMS, "--yes", "--overwrite"], scratch);

  // Copy the installed source over the previous snapshot. Only src/ is copied into
  // generated projects; dependencies.json beside it is read by the generator.
  fs.rmSync(path.join(output, "src"), { recursive: true, force: true });
  for (const dir of ["components", "hooks", "lib"]) {
    fs.cpSync(path.join(scratch, "src", dir), path.join(output, "src", dir), {
      recursive: true,
    });
  }
  // The stylesheet the shadcn CLI generated, including the theme tokens the
  // registry items merged in, becomes the React templates' stylesheet as is.
  for (const stylesheet of [
    "react-vite/src/index.css",
    "react-nextjs/src/app/globals.css",
  ]) {
    fs.copyFileSync(
      path.join(scratch, "src/index.css"),
      path.join(repoRoot, "src/pipecat/cli/templates/client", stylesheet),
    );
  }

  // Record where the snapshot came from and the dependency ranges the install
  // settled on. Ranges the shadcn CLI wrote are kept; packages seeded as "latest"
  // get a caret range on the version that was installed. `shadcn` is recorded as
  // a dependency rather than a devDependency because the stylesheet imports it at
  // build time.
  const pkg = JSON.parse(fs.readFileSync(path.join(scratch, "package.json"), "utf8"));
  const installedVersion = (name) =>
    JSON.parse(fs.readFileSync(path.join(scratch, "node_modules", name, "package.json"), "utf8"))
      .version;
  const dependencies = Object.fromEntries(
    Object.entries({ ...pkg.dependencies, shadcn: `^${installedVersion("shadcn")}` })
      .filter(([name]) => !TEMPLATE_OWNED.has(name) && !EXCLUDED.has(name))
      .map(([name, range]) => [name, range === "latest" ? `^${installedVersion(name)}` : range])
      .sort(([a], [b]) => a.localeCompare(b)),
  );
  fs.writeFileSync(
    path.join(output, "dependencies.json"),
    JSON.stringify({ source, shadcn: installedVersion("shadcn"), items: ITEMS, dependencies }, null, 2) +
      "\n",
  );
  succeeded = true;
  console.log(`\nSnapshot written to ${output}`);
} finally {
  registryServer?.server.close();
  if (succeeded && !options.keep) {
    fs.rmSync(scratch, { recursive: true, force: true });
  } else {
    console.log(`Scratch project left at ${scratch}`);
  }
}
