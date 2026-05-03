import React, { useRef, useState, useEffect, useCallback } from "react";
import type { CropRect } from "../api";

interface Props {
  imageUrl: string;
  onCropChange: (crop: CropRect | null) => void;
}

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export default function SelectionCanvas({ imageUrl, onCropChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  // Use refs for all drag state — avoids stale closures in event handlers
  const draggingRef = useRef(false);
  const startRef = useRef<{ x: number; y: number } | null>(null);
  const rectRef = useRef<Rect | null>(null);
  // State only for triggering re-render of the "Clear" button
  const [hasRect, setHasRect] = useState(false);
  const scaleRef = useRef({ sx: 1, sy: 1 });

  const draw = useCallback((selection: Rect | null) => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    if (selection) {
      ctx.strokeStyle = "#4A90D9";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.strokeRect(selection.x, selection.y, selection.w, selection.h);
      ctx.fillStyle = "rgba(74,144,217,0.15)";
      ctx.fillRect(selection.x, selection.y, selection.w, selection.h);
      ctx.setLineDash([]);
    }
  }, []);

  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const maxW = 600;
      const scale = Math.min(1, maxW / img.naturalWidth);
      canvas.width = img.naturalWidth * scale;
      canvas.height = img.naturalHeight * scale;
      scaleRef.current = {
        sx: img.naturalWidth / canvas.width,
        sy: img.naturalHeight / canvas.height,
      };
      draw(null);
    };
    img.src = imageUrl;
    rectRef.current = null;
    setHasRect(false);
    onCropChange(null);
  }, [imageUrl, draw, onCropChange]);

  // Account for CSS display size vs canvas pixel size (e.g. when canvas is scaled by CSS)
  const getPos = (e: React.MouseEvent<HTMLCanvasElement>): { x: number; y: number } => {
    const canvas = canvasRef.current!;
    const r = canvas.getBoundingClientRect();
    const scaleX = canvas.width / r.width;
    const scaleY = canvas.height / r.height;
    return {
      x: (e.clientX - r.left) * scaleX,
      y: (e.clientY - r.top) * scaleY,
    };
  };

  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = getPos(e);
    draggingRef.current = true;
    startRef.current = pos;
    rectRef.current = null;
    setHasRect(false);
    onCropChange(null);
    draw(null);
  };

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!draggingRef.current || !startRef.current) return;
    const pos = getPos(e);
    const s = startRef.current;
    const r: Rect = {
      x: Math.min(s.x, pos.x),
      y: Math.min(s.y, pos.y),
      w: Math.abs(pos.x - s.x),
      h: Math.abs(pos.y - s.y),
    };
    rectRef.current = r;
    draw(r);
  };

  const onMouseUp = () => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    const r = rectRef.current;
    if (r && r.w > 5 && r.h > 5) {
      setHasRect(true);
      const { sx, sy } = scaleRef.current;
      onCropChange({ x: r.x * sx, y: r.y * sy, w: r.w * sx, h: r.h * sy });
    } else {
      rectRef.current = null;
      setHasRect(false);
      draw(null);
      onCropChange(null);
    }
  };

  const clearSelection = () => {
    rectRef.current = null;
    setHasRect(false);
    draw(null);
    onCropChange(null);
  };

  return (
    <div>
      <p style={{ margin: "0 0 6px", fontSize: 13, color: "#555" }}>
        Drag to select a region (optional — leave empty to use full image)
      </p>
      <canvas
        ref={canvasRef}
        style={{ cursor: "crosshair", border: "1px solid #ddd", borderRadius: 8, display: "block", maxWidth: "100%" }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      />
      {hasRect && (
        <button
          onClick={clearSelection}
          style={{ marginTop: 6, fontSize: 12, padding: "3px 10px", cursor: "pointer" }}
        >
          Clear selection
        </button>
      )}
    </div>
  );
}
