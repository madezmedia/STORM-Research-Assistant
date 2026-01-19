'use client';

import { PageHeader } from '@/components/layout';
import { useAppStore } from '@/stores';
import { Settings, Monitor, Palette, Cpu, Server, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';

const models = [
  { value: 'openai/gpt-4o-mini', label: 'GPT-4o Mini', description: 'Fast and efficient' },
  { value: 'openai/gpt-4o', label: 'GPT-4o', description: 'Most capable' },
  { value: 'anthropic/claude-sonnet-4', label: 'Claude Sonnet 4', description: 'Balanced performance' },
  { value: 'xai/grok-4.1-fast-reasoning', label: 'Grok 4.1', description: 'Fast reasoning' },
];

export default function SettingsPage() {
  const {
    theme,
    defaultModel,
    sidebarCollapsed,
    isApiHealthy,
    setTheme,
    setDefaultModel,
    toggleSidebar,
    checkApiHealth,
  } = useAppStore();

  return (
    <div className="animate-fade-in max-w-4xl mx-auto">
      <PageHeader
        title="Settings"
        description="Configure your STORM Research Assistant"
      />

      <div className="space-y-6">
        {/* Appearance */}
        <div className="glass rounded-xl p-6">
          <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <Palette className="h-5 w-5 text-accent" />
            Appearance
          </h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-foreground mb-2">
                Theme
              </label>
              <div className="flex gap-3">
                {(['dark', 'light', 'system'] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setTheme(t)}
                    className={cn(
                      'flex-1 p-4 rounded-lg border-2 transition-all capitalize',
                      theme === t
                        ? 'border-primary bg-primary/10'
                        : 'border-border hover:border-primary/50'
                    )}
                  >
                    <Monitor className="h-5 w-5 mx-auto mb-2 text-foreground" />
                    <span className="text-sm text-foreground">{t}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="flex items-center justify-between p-4 rounded-lg bg-secondary/30">
              <div>
                <span className="font-medium text-foreground">Collapsed Sidebar</span>
                <p className="text-sm text-muted-foreground">Show compact sidebar navigation</p>
              </div>
              <button
                onClick={toggleSidebar}
                className={cn(
                  'relative w-12 h-6 rounded-full transition-colors',
                  sidebarCollapsed ? 'bg-primary' : 'bg-secondary'
                )}
              >
                <span
                  className={cn(
                    'absolute top-1 w-4 h-4 rounded-full bg-white transition-transform',
                    sidebarCollapsed ? 'left-7' : 'left-1'
                  )}
                />
              </button>
            </div>
          </div>
        </div>

        {/* AI Model */}
        <div className="glass rounded-xl p-6">
          <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <Cpu className="h-5 w-5 text-primary" />
            AI Model
          </h2>

          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Default Model
            </label>
            <div className="grid gap-3 md:grid-cols-2">
              {models.map((model) => (
                <button
                  key={model.value}
                  onClick={() => setDefaultModel(model.value)}
                  className={cn(
                    'p-4 rounded-lg border-2 text-left transition-all',
                    defaultModel === model.value
                      ? 'border-primary bg-primary/10'
                      : 'border-border hover:border-primary/50'
                  )}
                >
                  <span className="font-medium text-foreground">{model.label}</span>
                  <p className="text-sm text-muted-foreground mt-1">{model.description}</p>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* API Status */}
        <div className="glass rounded-xl p-6">
          <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <Server className="h-5 w-5 text-success" />
            API Configuration
          </h2>

          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 rounded-lg bg-secondary/30">
              <div className="flex items-center gap-3">
                <div className={cn(
                  'w-3 h-3 rounded-full',
                  isApiHealthy ? 'bg-success animate-pulse' : 'bg-destructive'
                )} />
                <div>
                  <span className="font-medium text-foreground">STORM API</span>
                  <p className="text-sm text-muted-foreground">
                    {isApiHealthy ? 'Connected and healthy' : 'Connection failed'}
                  </p>
                </div>
              </div>
              <button
                onClick={checkApiHealth}
                className="p-2 rounded-lg hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors"
                title="Refresh status"
              >
                <RefreshCw className="h-5 w-5" />
              </button>
            </div>

            <div className="p-4 rounded-lg bg-secondary/30">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-muted-foreground">API Endpoint</span>
              </div>
              <code className="text-sm text-foreground">
                {process.env.NEXT_PUBLIC_STORM_API_URL || '/api/v1 (via proxy)'}
              </code>
            </div>
          </div>
        </div>

        {/* About */}
        <div className="glass rounded-xl p-6">
          <h2 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <Settings className="h-5 w-5 text-muted-foreground" />
            About
          </h2>

          <div className="space-y-2 text-sm text-muted-foreground">
            <p><strong className="text-foreground">STORM Research Assistant</strong></p>
            <p>AI-powered content generation with multi-perspective STORM analysis</p>
            <p className="pt-2">
              Built with Next.js 16, React 19, and the Vercel AI Gateway
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
