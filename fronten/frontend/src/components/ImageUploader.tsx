import React, { useRef } from "react";

interface Props {
  onFile: (file: File, previewUrl: string) => void;
}

export default function ImageUploader({ onFile }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File) => {
    if (!file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    onFile(file, url);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <div
      onDrop={onDrop}
      onDragOver={(e) => e.preventDefault()}
      onClick={() => inputRef.current?.click()}
      style={{
        border: "2px dashed #4A90D9",
        borderRadius: 12,
        padding: "40px 20px",
        textAlign: "center",
        cursor: "pointer",
        background: "#F0F7FF",
        userSelect: "none",
      }}
    >
      <div style={{ fontSize: 48, marginBottom: 12 }}>🖼️</div>
      <p style={{ margin: 0, fontSize: 16, color: "#4A90D9", fontWeight: 600 }}>
        Click or drag an image here
      </p>
      <p style={{ margin: "6px 0 0", fontSize: 13, color: "#888" }}>
        Supports JPG, PNG, GIF
      </p>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/gif,image/webp"
        style={{ display: "none" }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleFile(f);
        }}
      />
    </div>
  );
}
