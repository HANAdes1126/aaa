#!/usr/bin/env node
/**
 * Regenerates ghost-preview.html from the current build output.
 *
 * The preview page needs the real `.meetly-ghost` rules to be faithful, but the
 * bundled stylesheet has a content hash in its filename. This script inlines the
 * ghost block straight from dist/assets so the preview stays in sync.
 *
 * Usage: npm run ghost:preview   (run after `npm run build`)
 */
import { readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const assetsDir = join(root, "dist", "assets");

let cssFile;
try {
  cssFile = readdirSync(assetsDir).find((name) => name.endsWith(".css"));
} catch {
  console.error("未找到 dist/assets —— 请先运行 npm run build");
  process.exit(1);
}
if (!cssFile) {
  console.error("dist/assets 下没有 CSS 产物 —— 请先运行 npm run build");
  process.exit(1);
}

const css = readFileSync(join(assetsDir, cssFile), "utf8");
const match = css.match(/\.meetly-ghost\s*\{/);
if (!match) {
  console.error("构建产物中没有 .meetly-ghost 规则，src/styles.css 可能被改动过");
  process.exit(1);
}

const template = readFileSync(join(root, "scripts", "ghost-preview.template.html"), "utf8");
const banner = `/* 由 scripts/build-ghost-preview.mjs 从 ${cssFile} 内联，请勿手工编辑 */\n`;
const html = template.replace("/* __APP_CSS__ */", banner + css.slice(match.index));

const target = join(root, "ghost-preview.html");
writeFileSync(target, html);

const rules = (css.slice(match.index).match(/meetly-ghost/g) ?? []).length;
console.log(`已生成 ghost-preview.html（内联 ${rules} 条 ghost 规则，来源 ${cssFile}）`);
