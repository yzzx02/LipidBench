import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const summaryDir = process.argv[2];
const previewDir = process.argv[3];
if (!summaryDir || !previewDir) {
  throw new Error("Usage: node build_cross_domain_workbook.mjs <summary-dir> <preview-dir>");
}

const inputs = [
  ["cross_domain_3seed_raw_metrics.csv", "Raw metrics"],
  ["cross_domain_3seed_mean_sd.csv", "Mean SD"],
  ["seen_vs_unseen_domain_gap.csv", "Domain gap"],
  ["domain_shift_descriptive_statistics.csv", "Domain stats"],
  ["domain_shift_attribute_statistics.csv", "Attribute stats"],
  ["external_B_failure_analysis.csv", "External B failure"],
  ["external_B_attribute_shift.csv", "External B attributes"],
];

const [firstFile, firstSheet] = inputs[0];
const firstCsv = await fs.readFile(path.join(summaryDir, firstFile), "utf8");
const workbook = await Workbook.fromCSV(firstCsv, { sheetName: firstSheet });
for (const [filename, sheetName] of inputs.slice(1)) {
  const csvText = await fs.readFile(path.join(summaryDir, filename), "utf8");
  await workbook.fromCSV(csvText, { sheetName });
}

const meanSheet = workbook.worksheets.getItem("Mean SD");
const meanValues = meanSheet.getUsedRange().values;
const meanHeader = meanValues[0].map((value) => String(value).replace(/^\uFEFF/, ""));
const coreColumns = [
  "condition",
  "det_f1_mean_sd",
  "mean_iou_mean_sd",
  "seed_ba_mean_sd",
  "seed_auroc_mean_sd",
  "seed_auprc_mean_sd",
];
const coreSheet = workbook.worksheets.add("Core results");
const coreRows = [
  ["Condition", "Detection F1", "Mean IoU", "Seed BA", "Seed AUROC", "Seed AUPRC"],
  ...meanValues.slice(1).filter((row) => row[0] != null).map((row) =>
    coreColumns.map((column) => row[meanHeader.indexOf(column)])
  ),
];
coreSheet.getRangeByIndexes(0, 0, coreRows.length, coreRows[0].length).values = coreRows;

const allSheetNames = [...inputs.map(([, sheetName]) => sheetName), "Core results"];
for (const sheetName of allSheetNames) {
  const sheet = workbook.worksheets.getItem(sheetName);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const used = sheet.getUsedRange();
  const typedValues = used.values;
  if (typeof typedValues[0][0] === "string") {
    typedValues[0][0] = typedValues[0][0].replace(/^\uFEFF/, "");
  }
  const numericPattern = /^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/;
  for (let rowIndex = 1; rowIndex < typedValues.length; rowIndex += 1) {
    for (let colIndex = 0; colIndex < typedValues[rowIndex].length; colIndex += 1) {
      const value = typedValues[rowIndex][colIndex];
      if (typeof value === "string" && numericPattern.test(value.trim())) {
        typedValues[rowIndex][colIndex] = Number(value);
      }
    }
  }
  used.values = typedValues;
  used.format.font = { name: "Arial", size: 10 };
  used.format.autofitColumns();
  used.format.autofitRows();
  const header = used.getRow(0);
  header.format = {
    fill: "#1F4E78",
    font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
    borders: { bottom: { style: "medium", color: "#17365D" } },
  };
  header.format.rowHeight = 30;
  used.format.borders = {
    insideHorizontal: { style: "thin", color: "#D9E2F3" },
    bottom: { style: "thin", color: "#9EADBC" },
  };
  used.getColumn(0).format.columnWidth = 20;
  if (used.columnCount > 1) used.getColumn(1).format.columnWidth = 24;
  if (used.columnCount > 2) used.getColumn(2).format.columnWidth = 24;
}

const integrity = JSON.parse(await fs.readFile(path.join(summaryDir, "integrity_audit.json"), "utf8"));
const auditSheet = workbook.worksheets.add("Integrity audit");
const auditRows = [["Check", "Value"]];
for (const [key, value] of Object.entries(integrity)) {
  if (key === "seed_checks") continue;
  auditRows.push([key, typeof value === "object" ? JSON.stringify(value) : value]);
}
for (const [seed, checks] of Object.entries(integrity.seed_checks)) {
  for (const [key, value] of Object.entries(checks)) {
    auditRows.push([`${seed}: ${key}`, value]);
  }
}
auditSheet.getRangeByIndexes(0, 0, auditRows.length, 2).values = auditRows;
auditSheet.showGridLines = false;
auditSheet.freezePanes.freezeRows(1);
auditSheet.getUsedRange().format.font = { name: "Arial", size: 10 };
auditSheet.getRange("A1:B1").format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
};
auditSheet.getRange(`A2:A${auditRows.length}`).format.columnWidth = 48;
auditSheet.getRange(`B2:B${auditRows.length}`).format.columnWidth = 24;

const inspection = await workbook.inspect({
  kind: "table",
  range: "Mean SD!A1:H8",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 8,
  maxChars: 5000,
});
console.log(inspection.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of allSheetNames) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  const safe = sheetName.toLowerCase().replaceAll(" ", "_");
  await fs.writeFile(path.join(previewDir, `${safe}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const auditPreview = await workbook.render({ sheetName: "Integrity audit", autoCrop: "all", scale: 1, format: "png" });
await fs.writeFile(path.join(previewDir, "integrity_audit.png"), new Uint8Array(await auditPreview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(summaryDir, "cross_domain_3seed_results.xlsx"));
console.log(JSON.stringify({ status: "completed", workbook: path.join(summaryDir, "cross_domain_3seed_results.xlsx") }));
