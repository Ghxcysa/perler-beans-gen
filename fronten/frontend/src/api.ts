const API = "http://localhost:8000";

export interface BeadColor {
  code: string;
  name: string;
  r: number;
  g: number;
  b: number;
}

export interface BeadCount extends BeadColor {
  count: number;
}

export interface Cell extends BeadColor {
  // same as BeadColor
}

export interface Step {
  step: number;
  row_start: number;
  row_end: number;
  cells: Array<{ row: number; col: number } & BeadColor>;
}

export interface ConvertResult {
  preview_b64: string;
  bead_list: BeadCount[];
  grid: Cell[][];
  steps: Step[];
  width: number;
  height: number;
}

export interface CropRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export async function convertImage(
  file: File,
  gridW: number,
  gridH: number,
  crop: CropRect | null,
  maxColors: number = 20,
): Promise<ConvertResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("grid_w", String(gridW));
  form.append("grid_h", String(gridH));
  form.append("max_colors", String(maxColors));
  if (crop) form.append("crop", JSON.stringify(crop));

  const res = await fetch(`${API}/convert`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function downloadPdf(
  file: File,
  gridW: number,
  gridH: number,
  crop: CropRect | null,
  maxColors: number = 20,
): Promise<void> {
  const form = new FormData();
  form.append("file", file);
  form.append("grid_w", String(gridW));
  form.append("grid_h", String(gridH));
  form.append("max_colors", String(maxColors));
  if (crop) form.append("crop", JSON.stringify(crop));

  const res = await fetch(`${API}/export-pdf`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "pattern.pdf";
  a.click();
  URL.revokeObjectURL(url);
}
