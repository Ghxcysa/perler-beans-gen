import type { BeadCount, ConvertResult, Step } from "../api";

interface Props {
  result: ConvertResult;
}

function rgb(r: number, g: number, b: number) {
  return `rgb(${r},${g},${b})`;
}

// ── 颜色图例 + 材料清单 ────────────────────────────────────
function BeadListTable({ list }: { list: BeadCount[] }) {
  return (
    <div style={{ marginTop: 16 }}>
      <h3 style={{ margin: "0 0 8px", fontSize: 15 }}>
        Color Legend &amp; Bead Count
      </h3>
      <div style={{ maxHeight: 300, overflowY: "auto", border: "1px solid #eee", borderRadius: 8 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ background: "#4A90D9", color: "#fff", position: "sticky", top: 0 }}>
              <th style={{ padding: "6px 10px", textAlign: "center" }}>Code</th>
              <th style={{ padding: "6px 10px", textAlign: "center" }}>Swatch</th>
              <th style={{ padding: "6px 10px", textAlign: "left" }}>Name</th>
              <th style={{ padding: "6px 10px", textAlign: "right" }}>Count</th>
            </tr>
          </thead>
          <tbody>
            {list.map((item, i) => (
              <tr key={item.code} style={{ background: i % 2 === 0 ? "#fff" : "#f5f8ff" }}>
                <td style={{ padding: "5px 10px", textAlign: "center", fontWeight: 700, fontFamily: "monospace", color: "#4A90D9", fontSize: 14 }}>
                  {item.code}
                </td>
                <td style={{ padding: "5px 10px", textAlign: "center" }}>
                  <span style={{
                    display: "inline-block",
                    width: 24, height: 24,
                    background: rgb(item.r, item.g, item.b),
                    border: "1px solid #ccc",
                    borderRadius: 4,
                    verticalAlign: "middle",
                  }} />
                </td>
                <td style={{ padding: "5px 10px" }}>{item.name}</td>
                <td style={{ padding: "5px 10px", textAlign: "right", fontWeight: 600 }}>{item.count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── 分步视图 ─────────────────────────────────────────────────
function StepsView({ steps, grid }: { steps: Step[]; grid: ConvertResult["grid"] }) {
  const CELL = 28;

  const renderStep = (step: Step) => {
    const W = grid[0].length;
    const activeRows = new Set<number>();
    for (let r = step.row_start; r <= step.row_end; r++) activeRows.add(r);

    // 统计本步用色
    const colorMap: Record<string, { code: string; name: string; r: number; g: number; b: number; count: number }> = {};
    grid.forEach((row, ri) => {
      if (!activeRows.has(ri)) return;
      row.forEach((c) => {
        if (!colorMap[c.code]) colorMap[c.code] = { ...c, count: 0 };
        colorMap[c.code].count++;
      });
    });

    return (
      <div key={step.step} style={{ marginBottom: 28 }}>
        <h4 style={{ margin: "0 0 6px", color: "#333", fontSize: 14 }}>
          Step {step.step} — Rows {step.row_start + 1}–{step.row_end + 1}
        </h4>

        {/* 本步颜色摘要 */}
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
          {Object.values(colorMap).map((c) => (
            <span key={c.code} style={{
              display: "inline-flex", alignItems: "center", gap: 4,
              background: "#f0f4ff", border: "1px solid #d0d8f0",
              borderRadius: 5, padding: "2px 8px", fontSize: 12,
            }}>
              <span style={{
                display: "inline-block", width: 12, height: 12,
                background: rgb(c.r, c.g, c.b), borderRadius: 2,
                border: "1px solid #aaa",
              }} />
              <b style={{ color: "#4A90D9" }}>{c.code}</b> ×{c.count}
            </span>
          ))}
        </div>

        {/* 网格：active 行显示编号，inactive 行淡化 */}
        <div style={{ overflowX: "auto" }}>
          <div style={{
            display: "inline-grid",
            gridTemplateColumns: `repeat(${W}, ${CELL}px)`,
            gap: 0,
            border: "1px solid #bbb",
          }}>
            {grid.map((row, ri) =>
              row.map((cell, ci) => {
                const active = activeRows.has(ri);
                const bgR = active ? cell.r : Math.round(cell.r * 0.25 + 220 * 0.75);
                const bgG = active ? cell.g : Math.round(cell.g * 0.25 + 220 * 0.75);
                const bgB = active ? cell.b : Math.round(cell.b * 0.25 + 220 * 0.75);
                const lum = 0.299 * bgR + 0.587 * bgG + 0.114 * bgB;
                const textColor = lum > 140 ? "#000" : "#fff";

                return (
                  <div
                    key={`${ri}-${ci}`}
                    title={`${cell.code} ${cell.name}`}
                    style={{
                      width: CELL, height: CELL,
                      background: rgb(bgR, bgG, bgB),
                      border: "0.5px solid rgba(0,0,0,0.15)",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontSize: 7, fontFamily: "monospace", fontWeight: 700,
                      color: textColor,
                      opacity: active ? 1 : 0.5,
                      boxSizing: "border-box",
                    }}
                  >
                    {cell.code}
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ marginTop: 24 }}>
      <h3 style={{ margin: "0 0 12px", fontSize: 15 }}>
        Step-by-Step Guide ({steps.length} steps, 3 rows each)
      </h3>
      {steps.map((s) => renderStep(s))}
    </div>
  );
}

// ── 主组件 ────────────────────────────────────────────────────
export default function GridPreview({ result }: Props) {
  return (
    <div>
      <h3 style={{ margin: "0 0 10px", fontSize: 15 }}>
        Pattern Preview — {result.width}×{result.height} ({result.width * result.height} beads · {result.bead_list.length} colors)
      </h3>

      {/* 后端渲染的带编号预览图（可横向滚动） */}
      <div style={{ overflowX: "auto", border: "1px solid #ddd", borderRadius: 8 }}>
        <img
          src={`data:image/png;base64,${result.preview_b64}`}
          alt="Bead pattern with color codes"
          style={{ display: "block", imageRendering: "pixelated" }}
        />
      </div>

      <BeadListTable list={result.bead_list} />
      <StepsView steps={result.steps} grid={result.grid} />
    </div>
  );
}
