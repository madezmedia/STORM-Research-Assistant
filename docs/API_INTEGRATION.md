# STORM Content Generation API Integration Guide

This guide explains how to integrate the STORM Content Generation API into your applications, blogging systems, and content management platforms.

## API Base URLs

| Environment | URL |
|-------------|-----|
| Production | `https://storm-fast-api.vercel.app` |
| Local Development | `http://localhost:8000` |

## Authentication

Currently, the API is open for internal use. For production deployments, add an API key header:

```bash
Authorization: Bearer your-api-key
```

## Quick Start

### 1. Generate Content (Simple)

The fastest way to generate content is using the `/api/v1/test-storm` endpoint:

```bash
curl -X POST https://storm-fast-api.vercel.app/api/v1/test-storm \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "5 Emerging Trends in Audio UGC Ads",
    "perspectives": 3
  }'
```

**Response:**
```json
{
  "success": true,
  "topic": "5 Emerging Trends in Audio UGC Ads",
  "perspectives": [
    "1. Industry Analyst's Perspective...",
    "2. Academic Researcher's Perspective...",
    "3. Practitioner's Perspective..."
  ],
  "outline": {
    "topic": "5 Emerging Trends in Audio UGC Ads",
    "content_type": "blog_post",
    "title": "Comprehensive Guide to 5 Emerging Trends in Audio UGC Ads",
    "sections": [...]
  },
  "sample_content": "# Full markdown article content..."
}
```

### 2. Full Content Generation Flow

For more control, use the briefs workflow:

```bash
# Step 1: Create a brief
BRIEF=$(curl -X POST https://storm-fast-api.vercel.app/api/v1/briefs \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Your Topic Here",
    "content_type": "blog_post",
    "word_count": 2000,
    "tone": "professional"
  }')

BRIEF_ID=$(echo $BRIEF | jq -r '.id')

# Step 2: Start generation
curl -X POST "https://storm-fast-api.vercel.app/api/v1/briefs/${BRIEF_ID}/generate"

# Step 3: Check status
curl "https://storm-fast-api.vercel.app/api/v1/briefs/${BRIEF_ID}/status"

# Step 4: Get generated content
curl "https://storm-fast-api.vercel.app/api/v1/content/${BRIEF_ID}"
```

---

## API Endpoints Reference

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy"
}
```

### Test LLM Connection

```
POST /api/v1/test-llm
```

**Request:**
```json
{
  "prompt": "Say hello!",
  "model": "openai/gpt-4o-mini"
}
```

**Response:**
```json
{
  "success": true,
  "response": "Hello! How can I help you today?",
  "model_used": "openai/gpt-4o-mini",
  "provider": "vercel-ai-gateway"
}
```

### STORM Analysis & Content Generation

```
POST /api/v1/test-storm
```

**Request:**
```json
{
  "topic": "Your research topic",
  "perspectives": 3
}
```

**Response:**
```json
{
  "success": true,
  "topic": "Your research topic",
  "perspectives": ["..."],
  "outline": {
    "topic": "Your research topic",
    "content_type": "blog_post",
    "title": "...",
    "sections": [
      {
        "title": "Introduction",
        "subsections": [{"title": "Overview"}, {"title": "Background"}]
      }
    ]
  },
  "sample_content": "# Full markdown content..."
}
```

### Content Briefs

#### Create Brief
```
POST /api/v1/briefs
```

**Request:**
```json
{
  "topic": "Your topic",
  "content_type": "blog_post",
  "word_count": 1500,
  "tone": "professional",
  "seo": {
    "primary_keyword": "main keyword",
    "secondary_keywords": ["kw1", "kw2"]
  },
  "target_audience": {
    "segment": "marketers",
    "expertise": "intermediate"
  }
}
```

#### List Briefs
```
GET /api/v1/briefs
```

#### Start Generation
```
POST /api/v1/briefs/{brief_id}/generate
```

#### Check Status
```
GET /api/v1/briefs/{brief_id}/status
```

**Response:**
```json
{
  "brief_id": "uuid",
  "status": "complete",
  "progress": 100,
  "current_phase": "Complete",
  "estimated_time_remaining": 0
}
```

#### Get Generated Content
```
GET /api/v1/content/{brief_id}
```

**Response:**
```json
{
  "id": "content-uuid",
  "brief_id": "brief-uuid",
  "title": "Article Title",
  "content": "# Full markdown content...",
  "word_count": 1523,
  "sections": [...],
  "seo_score": {
    "overall": 85,
    "readability": 90,
    "keyword_density": 80
  }
}
```

---

## Integration Examples

### JavaScript/TypeScript

```typescript
const STORM_API = 'https://storm-fast-api.vercel.app';

async function generateContent(topic: string): Promise<string> {
  const response = await fetch(`${STORM_API}/api/v1/test-storm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ topic, perspectives: 3 })
  });

  const data = await response.json();

  if (data.success) {
    return data.sample_content;
  }
  throw new Error('Content generation failed');
}

// Usage
const content = await generateContent('10 Tips for Better SEO');
console.log(content);
```

### Python

```python
import requests

STORM_API = "https://storm-fast-api.vercel.app"

def generate_content(topic: str, word_count: int = 1500) -> dict:
    """Generate content using STORM API"""

    # Create brief
    brief_response = requests.post(
        f"{STORM_API}/api/v1/briefs",
        json={
            "topic": topic,
            "content_type": "blog_post",
            "word_count": word_count,
            "tone": "professional"
        }
    )
    brief = brief_response.json()
    brief_id = brief["id"]

    # Start generation
    requests.post(f"{STORM_API}/api/v1/briefs/{brief_id}/generate")

    # Poll for completion
    import time
    while True:
        status = requests.get(f"{STORM_API}/api/v1/briefs/{brief_id}/status").json()
        if status["status"] == "complete":
            break
        elif status["status"] == "failed":
            raise Exception("Generation failed")
        time.sleep(2)

    # Get content
    content = requests.get(f"{STORM_API}/api/v1/content/{brief_id}").json()
    return content

# Usage
result = generate_content("AI in Healthcare")
print(result["content"])
```

### WordPress Integration

```php
<?php
function storm_generate_content($topic, $word_count = 1500) {
    $api_url = 'https://storm-fast-api.vercel.app';

    // Create brief
    $brief_response = wp_remote_post("$api_url/api/v1/briefs", array(
        'headers' => array('Content-Type' => 'application/json'),
        'body' => json_encode(array(
            'topic' => $topic,
            'content_type' => 'blog_post',
            'word_count' => $word_count,
            'tone' => 'professional'
        ))
    ));

    $brief = json_decode(wp_remote_retrieve_body($brief_response), true);
    $brief_id = $brief['id'];

    // Start generation
    wp_remote_post("$api_url/api/v1/briefs/$brief_id/generate");

    // Poll for completion (simplified)
    sleep(5);

    // Get content
    $content_response = wp_remote_get("$api_url/api/v1/content/$brief_id");
    $content = json_decode(wp_remote_retrieve_body($content_response), true);

    return $content;
}

// Usage in WordPress
$result = storm_generate_content('Top 10 Marketing Trends');
$post_content = $result['content'];
```

### Next.js API Route

```typescript
// pages/api/generate.ts
import type { NextApiRequest, NextApiResponse } from 'next';

const STORM_API = process.env.STORM_API_URL || 'https://storm-fast-api.vercel.app';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { topic } = req.body;

  try {
    const response = await fetch(`${STORM_API}/api/v1/test-storm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topic, perspectives: 3 })
    });

    const data = await response.json();
    res.status(200).json(data);
  } catch (error) {
    res.status(500).json({ error: 'Failed to generate content' });
  }
}
```

---

## Webhook Integration

For async workflows, you can set up webhooks to receive notifications when content generation is complete.

```bash
POST /api/v1/webhooks/generation-complete
```

Configure your webhook URL in your brief:
```json
{
  "topic": "...",
  "webhook_url": "https://your-app.com/webhook/storm-complete"
}
```

---

## Rate Limits

| Tier | Requests/Minute | Max Word Count |
|------|----------------|----------------|
| Free | 10 | 2000 |
| Pro | 60 | 5000 |
| Enterprise | Unlimited | Unlimited |

---

## Error Handling

All errors return JSON with an `error` field:

```json
{
  "detail": "Brief not found"
}
```

Common HTTP status codes:
- `200` - Success
- `201` - Created
- `400` - Bad request (invalid input)
- `404` - Resource not found
- `429` - Rate limited
- `500` - Server error

---

## Support

- GitHub Issues: https://github.com/madezmedia/STORM-Research-Assistant/issues
- Documentation: https://github.com/madezmedia/STORM-Research-Assistant
