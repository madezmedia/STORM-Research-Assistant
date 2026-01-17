"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";

export interface ContentBriefFormData {
  topic: string;
  contentType: string;
  seo: {
    primaryKeyword: string;
    secondaryKeywords: string[];
    targetVolume?: number;
    difficulty: "easy" | "medium" | "hard";
    intent: "informational" | "commercial" | "transactional";
  };
  geo: {
    enabled: boolean;
    location?: {
      country: string;
      state: string;
      city: string;
      zip: string;
    };
    localKeywords: string[];
    geoIntent: "local-service" | "regional-guide" | "national";
  };
  brandDirection: string;
  targetAudience: {
    segment: "ecommerce" | "local-business" | "agency" | "creator";
    painPoints: string[];
    expertise: "beginner" | "intermediate" | "advanced";
  };
  wordCount: number;
  tone: string;
  includeExamples: boolean;
  includeStats: boolean;
  includeLocalData: boolean;
}

interface ContentBriefFormProps {
  onSubmit: (data: ContentBriefFormData) => void;
  isSubmitting?: boolean;
}

export function ContentBriefForm({ onSubmit, isSubmitting = false }: ContentBriefFormProps) {
  const [formData, setFormData] = useState<ContentBriefFormData>({
    topic: "",
    contentType: "blog-post",
    seo: {
      primaryKeyword: "",
      secondaryKeywords: [],
      difficulty: "medium",
      intent: "informational",
    },
    geo: {
      enabled: false,
      localKeywords: [],
      geoIntent: "local-service",
    },
    brandDirection: "neutral",
    targetAudience: {
      segment: "ecommerce",
      painPoints: [],
      expertise: "intermediate",
    },
    wordCount: 2000,
    tone: "professional",
    includeExamples: true,
    includeStats: true,
    includeLocalData: false,
  });

  const [secondaryKeywordInput, setSecondaryKeywordInput] = useState("");
  const [painPointInput, setPainPointInput] = useState("");
  const [localKeywordInput, setLocalKeywordInput] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  const addSecondaryKeyword = () => {
    if (secondaryKeywordInput.trim()) {
      setFormData({
        ...formData,
        seo: {
          ...formData.seo,
          secondaryKeywords: [...formData.seo.secondaryKeywords, secondaryKeywordInput.trim()],
        },
      });
      setSecondaryKeywordInput("");
    }
  };

  const removeSecondaryKeyword = (index: number) => {
    setFormData({
      ...formData,
      seo: {
        ...formData.seo,
        secondaryKeywords: formData.seo.secondaryKeywords.filter((_, i) => i !== index),
      },
    });
  };

  const addPainPoint = () => {
    if (painPointInput.trim()) {
      setFormData({
        ...formData,
        targetAudience: {
          ...formData.targetAudience,
          painPoints: [...formData.targetAudience.painPoints, painPointInput.trim()],
        },
      });
      setPainPointInput("");
    }
  };

  const removePainPoint = (index: number) => {
    setFormData({
      ...formData,
      targetAudience: {
        ...formData.targetAudience,
        painPoints: formData.targetAudience.painPoints.filter((_, i) => i !== index),
      },
    });
  };

  const addLocalKeyword = () => {
    if (localKeywordInput.trim()) {
      setFormData({
        ...formData,
        geo: {
          ...formData.geo,
          localKeywords: [...formData.geo.localKeywords, localKeywordInput.trim()],
        },
      });
      setLocalKeywordInput("");
    }
  };

  const removeLocalKeyword = (index: number) => {
    setFormData({
      ...formData,
      geo: {
        ...formData.geo,
        localKeywords: formData.geo.localKeywords.filter((_, i) => i !== index),
      },
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Content Brief</CardTitle>
          <CardDescription>Define your content requirements</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="topic">Topic</Label>
            <Input
              id="topic"
              value={formData.topic}
              onChange={(e) => setFormData({ ...formData, topic: e.target.value })}
              placeholder="What do you want to write about?"
              required
              disabled={isSubmitting}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="contentType">Content Type</Label>
            <Select
              value={formData.contentType}
              onValueChange={(value) => setFormData({ ...formData, contentType: value })}
              disabled={isSubmitting}
            >
              <SelectTrigger id="contentType">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="blog-post">Blog Post</SelectItem>
                <SelectItem value="article">Article</SelectItem>
                <SelectItem value="guide">Guide</SelectItem>
                <SelectItem value="landing-page">Landing Page</SelectItem>
                <SelectItem value="product-description">Product Description</SelectItem>
                <SelectItem value="case-study">Case Study</SelectItem>
                <SelectItem value="whitepaper">Whitepaper</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="wordCount">Word Count: {formData.wordCount}</Label>
            <Input
              id="wordCount"
              type="range"
              min="1000"
              max="10000"
              step="500"
              value={formData.wordCount}
              onChange={(e) => setFormData({ ...formData, wordCount: parseInt(e.target.value) })}
              disabled={isSubmitting}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="tone">Tone</Label>
            <Select
              value={formData.tone}
              onValueChange={(value) => setFormData({ ...formData, tone: value })}
              disabled={isSubmitting}
            >
              <SelectTrigger id="tone">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="professional">Professional</SelectItem>
                <SelectItem value="casual">Casual</SelectItem>
                <SelectItem value="friendly">Friendly</SelectItem>
                <SelectItem value="authoritative">Authoritative</SelectItem>
                <SelectItem value="conversational">Conversational</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="brandDirection">Brand Direction</Label>
            <Select
              value={formData.brandDirection}
              onValueChange={(value) => setFormData({ ...formData, brandDirection: value })}
              disabled={isSubmitting}
            >
              <SelectTrigger id="brandDirection">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="neutral">Neutral</SelectItem>
                <SelectItem value="bold">Bold</SelectItem>
                <SelectItem value="conservative">Conservative</SelectItem>
                <SelectItem value="innovative">Innovative</SelectItem>
                <SelectItem value="traditional">Traditional</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="seo" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="seo">SEO</TabsTrigger>
          <TabsTrigger value="audience">Audience</TabsTrigger>
          <TabsTrigger value="geo">GEO</TabsTrigger>
        </TabsList>

        <TabsContent value="seo">
          <Card>
            <CardHeader>
              <CardTitle>SEO Settings</CardTitle>
              <CardDescription>Optimize your content for search engines</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="primaryKeyword">Primary Keyword</Label>
                <Input
                  id="primaryKeyword"
                  value={formData.seo.primaryKeyword}
                  onChange={(e) => setFormData({
                    ...formData,
                    seo: { ...formData.seo, primaryKeyword: e.target.value }
                  })}
                  placeholder="Main keyword for your content"
                  required
                  disabled={isSubmitting}
                />
              </div>

              <div className="space-y-2">
                <Label>Secondary Keywords</Label>
                <div className="flex gap-2">
                  <Input
                    value={secondaryKeywordInput}
                    onChange={(e) => setSecondaryKeywordInput(e.target.value)}
                    placeholder="Add secondary keyword"
                    onKeyPress={(e) => e.key === "Enter" && (e.preventDefault(), addSecondaryKeyword())}
                    disabled={isSubmitting}
                  />
                  <Button type="button" onClick={addSecondaryKeyword} disabled={isSubmitting}>
                    Add
                  </Button>
                </div>
                <div className="flex flex-wrap gap-2 mt-2">
                  {formData.seo.secondaryKeywords.map((keyword, index) => (
                    <Badge key={index} variant="secondary">
                      {keyword}
                      <button
                        type="button"
                        onClick={() => removeSecondaryKeyword(index)}
                        className="ml-2 hover:text-red-500"
                        disabled={isSubmitting}
                      >
                        ×
                      </button>
                    </Badge>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="difficulty">Difficulty</Label>
                  <Select
                    value={formData.seo.difficulty}
                    onValueChange={(value: "easy" | "medium" | "hard") => setFormData({
                      ...formData,
                      seo: { ...formData.seo, difficulty: value }
                    })}
                    disabled={isSubmitting}
                  >
                    <SelectTrigger id="difficulty">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="easy">Easy</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="hard">Hard</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="intent">Intent</Label>
                  <Select
                    value={formData.seo.intent}
                    onValueChange={(value: "informational" | "commercial" | "transactional") => setFormData({
                      ...formData,
                      seo: { ...formData.seo, intent: value }
                    })}
                    disabled={isSubmitting}
                  >
                    <SelectTrigger id="intent">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="informational">Informational</SelectItem>
                      <SelectItem value="commercial">Commercial</SelectItem>
                      <SelectItem value="transactional">Transactional</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="audience">
          <Card>
            <CardHeader>
              <CardTitle>Target Audience</CardTitle>
              <CardDescription>Define who you're writing for</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="segment">Segment</Label>
                <Select
                  value={formData.targetAudience.segment}
                  onValueChange={(value: "ecommerce" | "local-business" | "agency" | "creator") => setFormData({
                    ...formData,
                    targetAudience: { ...formData.targetAudience, segment: value }
                  })}
                  disabled={isSubmitting}
                >
                  <SelectTrigger id="segment">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ecommerce">E-commerce</SelectItem>
                    <SelectItem value="local-business">Local Business</SelectItem>
                    <SelectItem value="agency">Agency</SelectItem>
                    <SelectItem value="creator">Content Creator</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="expertise">Expertise Level</Label>
                <Select
                  value={formData.targetAudience.expertise}
                  onValueChange={(value: "beginner" | "intermediate" | "advanced") => setFormData({
                    ...formData,
                    targetAudience: { ...formData.targetAudience, expertise: value }
                  })}
                  disabled={isSubmitting}
                >
                  <SelectTrigger id="expertise">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="beginner">Beginner</SelectItem>
                    <SelectItem value="intermediate">Intermediate</SelectItem>
                    <SelectItem value="advanced">Advanced</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Pain Points</Label>
                <div className="flex gap-2">
                  <Input
                    value={painPointInput}
                    onChange={(e) => setPainPointInput(e.target.value)}
                    placeholder="Add pain point"
                    onKeyPress={(e) => e.key === "Enter" && (e.preventDefault(), addPainPoint())}
                    disabled={isSubmitting}
                  />
                  <Button type="button" onClick={addPainPoint} disabled={isSubmitting}>
                    Add
                  </Button>
                </div>
                <div className="flex flex-wrap gap-2 mt-2">
                  {formData.targetAudience.painPoints.map((point, index) => (
                    <Badge key={index} variant="secondary">
                      {point}
                      <button
                        type="button"
                        onClick={() => removePainPoint(index)}
                        className="ml-2 hover:text-red-500"
                        disabled={isSubmitting}
                      >
                        ×
                      </button>
                    </Badge>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="geo">
          <Card>
            <CardHeader>
              <CardTitle>GEO Targeting</CardTitle>
              <CardDescription>Optimize content for local search</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <Label htmlFor="geoEnabled">Enable GEO Targeting</Label>
                <Switch
                  id="geoEnabled"
                  checked={formData.geo.enabled}
                  onCheckedChange={(checked) => setFormData({
                    ...formData,
                    geo: { ...formData.geo, enabled: checked }
                  })}
                  disabled={isSubmitting}
                />
              </div>

              {formData.geo.enabled && (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="country">Country</Label>
                      <Input
                        id="country"
                        value={formData.geo.location?.country || ""}
                        onChange={(e) => setFormData({
                          ...formData,
                          geo: {
                            ...formData.geo,
                            location: {
                              country: e.target.value,
                              state: formData.geo.location?.state || '',
                              city: formData.geo.location?.city || '',
                              zip: formData.geo.location?.zip || ''
                            }
                          }
                        })}
                        placeholder="Country"
                        disabled={isSubmitting}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="state">State</Label>
                      <Input
                        id="state"
                        value={formData.geo.location?.state || ""}
                        onChange={(e) => setFormData({
                          ...formData,
                          geo: {
                            ...formData.geo,
                            location: {
                              country: formData.geo.location?.country || '',
                              state: e.target.value,
                              city: formData.geo.location?.city || '',
                              zip: formData.geo.location?.zip || ''
                            }
                          }
                        })}
                        placeholder="State"
                        disabled={isSubmitting}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="city">City</Label>
                      <Input
                        id="city"
                        value={formData.geo.location?.city || ""}
                        onChange={(e) => setFormData({
                          ...formData,
                          geo: {
                            ...formData.geo,
                            location: {
                              country: formData.geo.location?.country || '',
                              state: formData.geo.location?.state || '',
                              city: e.target.value,
                              zip: formData.geo.location?.zip || ''
                            }
                          }
                        })}
                        placeholder="City"
                        disabled={isSubmitting}
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="zip">ZIP Code</Label>
                      <Input
                        id="zip"
                        value={formData.geo.location?.zip || ""}
                        onChange={(e) => setFormData({
                          ...formData,
                          geo: {
                            ...formData.geo,
                            location: {
                              country: formData.geo.location?.country || '',
                              state: formData.geo.location?.state || '',
                              city: formData.geo.location?.city || '',
                              zip: e.target.value
                            }
                          }
                        })}
                        placeholder="ZIP Code"
                        disabled={isSubmitting}
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="geoIntent">GEO Intent</Label>
                    <Select
                      value={formData.geo.geoIntent}
                      onValueChange={(value: "local-service" | "regional-guide" | "national") => setFormData({
                        ...formData,
                        geo: { ...formData.geo, geoIntent: value }
                      })}
                      disabled={isSubmitting}
                    >
                      <SelectTrigger id="geoIntent">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="local-service">Local Service</SelectItem>
                        <SelectItem value="regional-guide">Regional Guide</SelectItem>
                        <SelectItem value="national">National</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Local Keywords</Label>
                    <div className="flex gap-2">
                      <Input
                        value={localKeywordInput}
                        onChange={(e) => setLocalKeywordInput(e.target.value)}
                        placeholder="Add local keyword"
                        onKeyPress={(e) => e.key === "Enter" && (e.preventDefault(), addLocalKeyword())}
                        disabled={isSubmitting}
                      />
                      <Button type="button" onClick={addLocalKeyword} disabled={isSubmitting}>
                        Add
                      </Button>
                    </div>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {formData.geo.localKeywords.map((keyword, index) => (
                        <Badge key={index} variant="secondary">
                          {keyword}
                          <button
                            type="button"
                            onClick={() => removeLocalKeyword(index)}
                            className="ml-2 hover:text-red-500"
                            disabled={isSubmitting}
                          >
                            ×
                          </button>
                        </Badge>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <Card>
        <CardHeader>
          <CardTitle>Content Options</CardTitle>
          <CardDescription>Additional content generation settings</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="includeExamples">Include Examples</Label>
            <Switch
              id="includeExamples"
              checked={formData.includeExamples}
              onCheckedChange={(checked) => setFormData({ ...formData, includeExamples: checked })}
              disabled={isSubmitting}
            />
          </div>

          <div className="flex items-center justify-between">
            <Label htmlFor="includeStats">Include Statistics</Label>
            <Switch
              id="includeStats"
              checked={formData.includeStats}
              onCheckedChange={(checked) => setFormData({ ...formData, includeStats: checked })}
              disabled={isSubmitting}
            />
          </div>

          <div className="flex items-center justify-between">
            <Label htmlFor="includeLocalData">Include Local Data</Label>
            <Switch
              id="includeLocalData"
              checked={formData.includeLocalData}
              onCheckedChange={(checked) => setFormData({ ...formData, includeLocalData: checked })}
              disabled={isSubmitting}
            />
          </div>
        </CardContent>
      </Card>

      <Button type="submit" className="w-full" disabled={isSubmitting}>
        {isSubmitting ? "Creating Brief..." : "Create Brief"}
      </Button>
    </form>
  );
}
