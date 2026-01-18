"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface SlideshowConfig {
  content: string;
  max_slides: number;
  style: "professional" | "minimalist" | "vibrant";
  autoplay: boolean;
  duration: number;
  transition: "fade" | "slide" | "zoom";
  theme: "dark" | "light";
  show_controls: boolean;
  loop: boolean;
  generate_images: boolean;
  title: string;
  brand_colors?: {
    primary: string;
    secondary: string;
  };
}

interface SlideshowGeneratorProps {
  capabilities: {
    modules_available: boolean;
    ffmpeg: { available: boolean; version?: string };
    zai_configured: boolean;
    gateway_configured: boolean;
    supported_formats: string[];
  } | null;
}

export function SlideshowGenerator({ capabilities }: SlideshowGeneratorProps) {
  const [config, setConfig] = useState<SlideshowConfig>({
    content: "",
    max_slides: 8,
    style: "professional",
    autoplay: true,
    duration: 5,
    transition: "fade",
    theme: "dark",
    show_controls: true,
    loop: true,
    generate_images: true,
    title: "Presentation",
  });

  const [isGenerating, setIsGenerating] = useState(false);
  const [result, setResult] = useState<{
    success: boolean;
    job_id?: string;
    slide_count?: number;
    html?: string;
    embed_code?: string;
    error?: string;
  } | null>(null);
  const [activeTab, setActiveTab] = useState("paste");

  const handleGenerate = async () => {
    if (!config.content.trim()) {
      alert("Please enter some content");
      return;
    }

    setIsGenerating(true);
    setResult(null);

    try {
      const response = await fetch("/api/v1/slides/slideshow", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ success: false, error: String(error) });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownloadHTML = () => {
    if (!result?.html) return;
    const blob = new Blob([result.html], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${config.title.replace(/\s+/g, "_")}.html`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleCopyEmbed = () => {
    if (!result?.embed_code) return;
    navigator.clipboard.writeText(result.embed_code);
    alert("Embed code copied to clipboard!");
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Left Column - Input & Options */}
      <div className="space-y-6">
        {/* Content Input */}
        <Card>
          <CardHeader>
            <CardTitle>Content</CardTitle>
            <CardDescription>
              Enter or paste the content to transform into slides
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="mb-4">
                <TabsTrigger value="paste">Paste Content</TabsTrigger>
                <TabsTrigger value="storm">From STORM</TabsTrigger>
              </TabsList>
              <TabsContent value="paste">
                <Textarea
                  placeholder="Paste your article, blog post, or content here..."
                  className="min-h-[200px] font-mono text-sm"
                  value={config.content}
                  onChange={(e) => setConfig({ ...config, content: e.target.value })}
                />
              </TabsContent>
              <TabsContent value="storm">
                <p className="text-sm text-slate-500 mb-4">
                  Select previously generated STORM content
                </p>
                <Select>
                  <SelectTrigger>
                    <SelectValue placeholder="Select content brief..." />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">No content available</SelectItem>
                  </SelectContent>
                </Select>
              </TabsContent>
            </Tabs>

            <div className="mt-4">
              <Label htmlFor="title">Presentation Title</Label>
              <Input
                id="title"
                value={config.title}
                onChange={(e) => setConfig({ ...config, title: e.target.value })}
                placeholder="Enter presentation title"
                className="mt-1"
              />
            </div>
          </CardContent>
        </Card>

        {/* Style & Options */}
        <Card>
          <CardHeader>
            <CardTitle>Style & Options</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Style Selection */}
            <div>
              <Label>Visual Style</Label>
              <div className="grid grid-cols-3 gap-3 mt-2">
                {["professional", "minimalist", "vibrant"].map((style) => (
                  <button
                    key={style}
                    onClick={() => setConfig({ ...config, style: style as any })}
                    className={`p-3 rounded-lg border-2 text-sm font-medium capitalize transition-all ${
                      config.style === style
                        ? "border-indigo-500 bg-indigo-50 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300"
                        : "border-slate-200 hover:border-slate-300 dark:border-slate-700"
                    }`}
                  >
                    {style}
                  </button>
                ))}
              </div>
            </div>

            {/* Slides Count */}
            <div>
              <Label htmlFor="max_slides">Number of Slides: {config.max_slides}</Label>
              <input
                type="range"
                id="max_slides"
                min="3"
                max="15"
                value={config.max_slides}
                onChange={(e) => setConfig({ ...config, max_slides: parseInt(e.target.value) })}
                className="w-full mt-2"
              />
            </div>

            {/* Duration */}
            <div>
              <Label htmlFor="duration">Slide Duration: {config.duration}s</Label>
              <input
                type="range"
                id="duration"
                min="3"
                max="15"
                value={config.duration}
                onChange={(e) => setConfig({ ...config, duration: parseInt(e.target.value) })}
                className="w-full mt-2"
              />
            </div>

            {/* Transition */}
            <div>
              <Label>Transition Effect</Label>
              <Select
                value={config.transition}
                onValueChange={(v) => setConfig({ ...config, transition: v as any })}
              >
                <SelectTrigger className="mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="fade">Fade</SelectItem>
                  <SelectItem value="slide">Slide</SelectItem>
                  <SelectItem value="zoom">Zoom</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Toggles */}
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center justify-between">
                <Label htmlFor="autoplay">Autoplay</Label>
                <Switch
                  id="autoplay"
                  checked={config.autoplay}
                  onCheckedChange={(v) => setConfig({ ...config, autoplay: v })}
                />
              </div>
              <div className="flex items-center justify-between">
                <Label htmlFor="loop">Loop</Label>
                <Switch
                  id="loop"
                  checked={config.loop}
                  onCheckedChange={(v) => setConfig({ ...config, loop: v })}
                />
              </div>
              <div className="flex items-center justify-between">
                <Label htmlFor="controls">Controls</Label>
                <Switch
                  id="controls"
                  checked={config.show_controls}
                  onCheckedChange={(v) => setConfig({ ...config, show_controls: v })}
                />
              </div>
              <div className="flex items-center justify-between">
                <Label htmlFor="images">AI Images</Label>
                <Switch
                  id="images"
                  checked={config.generate_images}
                  onCheckedChange={(v) => setConfig({ ...config, generate_images: v })}
                />
              </div>
            </div>

            {/* Theme */}
            <div className="flex items-center justify-between">
              <Label>Theme</Label>
              <div className="flex gap-2">
                <button
                  onClick={() => setConfig({ ...config, theme: "dark" })}
                  className={`px-4 py-2 rounded-lg text-sm ${
                    config.theme === "dark"
                      ? "bg-slate-800 text-white"
                      : "bg-slate-200 text-slate-600"
                  }`}
                >
                  Dark
                </button>
                <button
                  onClick={() => setConfig({ ...config, theme: "light" })}
                  className={`px-4 py-2 rounded-lg text-sm ${
                    config.theme === "light"
                      ? "bg-white border-2 border-slate-300 text-slate-800"
                      : "bg-slate-200 text-slate-600"
                  }`}
                >
                  Light
                </button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Generate Button */}
        <Button
          onClick={handleGenerate}
          disabled={isGenerating || !config.content.trim()}
          className="w-full h-12 text-lg bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
        >
          {isGenerating ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Generating Slideshow...
            </span>
          ) : (
            "Generate Slideshow"
          )}
        </Button>
      </div>

      {/* Right Column - Preview & Download */}
      <div className="space-y-6">
        <Card className="h-full">
          <CardHeader>
            <CardTitle>Preview</CardTitle>
            <CardDescription>
              {result?.success
                ? `${result.slide_count} slides generated`
                : "Your slideshow will appear here"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {result?.success && result.job_id ? (
              <div className="space-y-4">
                {/* Preview iframe */}
                <div className="aspect-video bg-slate-900 rounded-lg overflow-hidden">
                  <iframe
                    src={`/api/v1/slides/${result.job_id}/embed`}
                    className="w-full h-full"
                    title="Slideshow Preview"
                  />
                </div>

                {/* Actions */}
                <div className="flex gap-3">
                  <Button onClick={handleDownloadHTML} className="flex-1">
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    Download HTML
                  </Button>
                  <Button onClick={handleCopyEmbed} variant="outline" className="flex-1">
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                    Copy Embed
                  </Button>
                </div>

                {/* Embed Code */}
                {result.embed_code && (
                  <div className="mt-4">
                    <Label>Embed Code</Label>
                    <pre className="mt-2 p-3 bg-slate-100 dark:bg-slate-800 rounded-lg text-xs overflow-x-auto">
                      {result.embed_code}
                    </pre>
                  </div>
                )}
              </div>
            ) : result?.error ? (
              <div className="text-center py-12">
                <div className="text-red-500 mb-2">Generation failed</div>
                <p className="text-sm text-slate-500">{result.error}</p>
              </div>
            ) : (
              <div className="aspect-video bg-slate-100 dark:bg-slate-800 rounded-lg flex items-center justify-center">
                <div className="text-center text-slate-400">
                  <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M7 4v16M17 4v16M3 8h4m10 0h4M3 12h18M3 16h4m10 0h4M4 20h16a1 1 0 001-1V5a1 1 0 00-1-1H4a1 1 0 00-1 1v14a1 1 0 001 1z" />
                  </svg>
                  <p>Enter content and click Generate</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
