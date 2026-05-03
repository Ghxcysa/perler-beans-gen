import { useState, useCallback } from "react";
import ImageUploader from "./components/ImageUploader";
import SelectionCanvas from "./components/SelectionCanvas";
import GridPreview from "./components/GridPreview";
import { convertImage, downloadPdf } from "./api";
import type { ConvertResult, CropRect } from "./api";

const GRID_SIZES = [29, 48, 58];

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [crop, setCrop] = useState<CropRect | null>(null);
  const [gridSize, setGridSize] = useState<number>(29);
  const [maxColors, setMaxColors] = useState<number>(20);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ConvertResult | null>(null);
  const [pdfLoading, setPdfLoading] = useState(false);

  const handleFile = useCallback((f: File, url: string) => {
    setFile(f);
    setPreviewUrl(url);
    setResult(null);
    setError(null);
    setCrop(null);
  }, []);

  const handleConvert = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await convertImage(file, gridSize, gridSize, crop, maxColors);
      setResult(res);
    } catch (e: unknown) {
      setError(String((e as { message?: string })?.message ?? e));
    } finally {
      setLoading(false);
    }
  };

  const handleExportPdf = async () => {
    if (!file) return;
    setPdfLoading(true);
    try {
      await downloadPdf(file, gridSize, gridSize, crop, maxColors);
    } catch (e: unknown) {
      setError(String((e as { message?: string })?.message ?? e));
    } finally {
      setPdfLoading(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: "#F8FAFC", fontFamily: "system-ui, sans-serif" }}>
      {/* Header */}
      <header style={{ background: "#4A90D9", color: "#fff", padding: "16px 32px", boxShadow: "0 2px 8px rgba(0,0,0,0.1)" }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>🎨 Creative PinDou</h1>
        <p style={{ margin: "2px 0 0", fontSize: 13, opacity: 0.85 }}>
          Turn any image into a printable Perler bead pattern
        </p>
      </header>

      <main style={{ maxWidth: 900, margin: "0 auto", padding: "24px 16px", display: "flex", flexDirection: "column", gap: 24 }}>

        {/* Step 1: Upload */}
        <section style={cardStyle}>
          <SectionTitle num={1} text="Upload Image" />
          <ImageUploader onFile={handleFile} />
        </section>

        {/* Step 2: Select Region */}
        {previewUrl && (
          <section style={cardStyle}>
            <SectionTitle num={2} text="Select Target Area (optional)" />
            <SelectionCanvas imageUrl={previewUrl} onCropChange={setCrop} />
            {crop && (
              <p style={{ margin: "6px 0 0", fontSize: 12, color: "#4A90D9" }}>
                ✅ Selection: {Math.round(crop.w)}×{Math.round(crop.h)} px
              </p>
            )}
          </section>
        )}

        {/* Step 3: Settings + Convert */}
        {file && (
          <section style={cardStyle}>
            <SectionTitle num={3} text="Grid Size & Convert" />
            <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 16 }}>
              <label style={{ fontSize: 14, fontWeight: 500 }}>Grid size:</label>
              {GRID_SIZES.map((s) => (
                <button
                  key={s}
                  onClick={() => setGridSize(s)}
                  style={{
                    ...toggleBtn,
                    background: gridSize === s ? "#4A90D9" : "#fff",
                    color: gridSize === s ? "#fff" : "#4A90D9",
                  }}
                >
                  {s}×{s}
                </button>
              ))}
            </div>

            {/* Max colors slider */}
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
              <label style={{ fontSize: 14, fontWeight: 500, whiteSpace: "nowrap" }}>
                Max colors:
              </label>
              <input
                type="range"
                min={5}
                max={40}
                step={1}
                value={maxColors}
                onChange={(e) => setMaxColors(Number(e.target.value))}
                style={{ width: 180, accentColor: "#4A90D9" }}
              />
              <span style={{
                minWidth: 32, textAlign: "center", fontWeight: 700,
                color: "#4A90D9", fontSize: 15,
              }}>
                {maxColors}
              </span>
              <span style={{ fontSize: 12, color: "#888" }}>
                (fewer = cleaner, more = more detailed)
              </span>
            </div>

            <div style={{ display: "flex", alignItems: "center" }}>
              <button
                onClick={handleConvert}
                disabled={loading}
                style={{
                  padding: "10px 28px",
                  background: loading ? "#aaa" : "#4A90D9",
                  color: "#fff",
                  border: "none",
                  borderRadius: 8,
                  fontWeight: 700,
                  fontSize: 15,
                  cursor: loading ? "not-allowed" : "pointer",
                }}
              >
                {loading ? "⏳ Converting…" : "▶ Convert"}
              </button>
            </div>

            {error && (
              <p style={{ marginTop: 8, color: "#c00", fontSize: 13 }}>⚠ {error}</p>
            )}
          </section>
        )}

        {/* Step 4: Preview + Export */}
        {result && (
          <section style={cardStyle}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <SectionTitle num={4} text="Preview & Export" />
              <button
                onClick={handleExportPdf}
                disabled={pdfLoading}
                style={{
                  padding: "8px 20px",
                  background: pdfLoading ? "#aaa" : "#E67E22",
                  color: "#fff",
                  border: "none",
                  borderRadius: 8,
                  fontWeight: 700,
                  fontSize: 14,
                  cursor: pdfLoading ? "not-allowed" : "pointer",
                }}
              >
                {pdfLoading ? "⏳ Generating…" : "📄 Export PDF"}
              </button>
            </div>
            <GridPreview result={result} />
          </section>
        )}
      </main>
    </div>
  );
}

function SectionTitle({ num, text }: { num: number; text: string }) {
  return (
    <h2 style={{ margin: "0 0 14px", fontSize: 16, fontWeight: 700, color: "#333", display: "flex", alignItems: "center", gap: 8 }}>
      <span style={{
        background: "#4A90D9", color: "#fff", borderRadius: "50%",
        width: 24, height: 24, display: "inline-flex", alignItems: "center",
        justifyContent: "center", fontSize: 13, flexShrink: 0,
      }}>
        {num}
      </span>
      {text}
    </h2>
  );
}

const cardStyle: React.CSSProperties = {
  background: "#fff",
  borderRadius: 12,
  padding: 20,
  boxShadow: "0 1px 6px rgba(0,0,0,0.07)",
};

const toggleBtn: React.CSSProperties = {
  padding: "6px 16px",
  border: "1.5px solid #4A90D9",
  borderRadius: 6,
  fontWeight: 600,
  fontSize: 14,
  cursor: "pointer",
};
