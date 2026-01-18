"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SlideshowGenerator } from "@/components/slideshow/SlideshowGenerator";

export default function SlideshowPage() {
  const [capabilities, setCapabilities] = useState<{
    modules_available: boolean;
    ffmpeg: { available: boolean; version?: string };
    zai_configured: boolean;
    gateway_configured: boolean;
    supported_formats: string[];
  } | null>(null);

  useEffect(() => {
    // Check API capabilities
    fetch("/api/v1/slides/status")
      .then((res) => res.json())
      .then(setCapabilities)
      .catch(console.error);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
            Slideshow Generator
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-2">
            Transform your content into stunning presentations with AI-generated visuals
          </p>
        </div>

        {/* Status Banner */}
        {capabilities && (
          <Card className="mb-6 border-indigo-200 dark:border-indigo-800">
            <CardContent className="py-3">
              <div className="flex items-center gap-4 text-sm">
                <span className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${capabilities.modules_available ? "bg-green-500" : "bg-red-500"}`} />
                  API
                </span>
                <span className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${capabilities.zai_configured ? "bg-green-500" : "bg-yellow-500"}`} />
                  Z.AI Images
                </span>
                <span className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${capabilities.ffmpeg.available ? "bg-green-500" : "bg-yellow-500"}`} />
                  Video Export
                </span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Main Generator */}
        <SlideshowGenerator
          capabilities={capabilities}
        />
      </div>
    </div>
  );
}
