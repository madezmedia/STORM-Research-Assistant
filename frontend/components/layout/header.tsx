'use client';

import { useAppStore } from '@/stores';
import { cn } from '@/lib/utils';
import { Bell, Search, User } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface HeaderProps {
  title?: string;
  description?: string;
  actions?: React.ReactNode;
}

export function Header({ title, description, actions }: HeaderProps) {
  const { sidebarCollapsed, isApiHealthy } = useAppStore();

  return (
    <header
      className={cn(
        'fixed top-0 right-0 z-30 h-16 bg-background/80 backdrop-blur-xl border-b border-border/50 flex items-center justify-between px-6 transition-all duration-300',
        sidebarCollapsed ? 'left-16' : 'left-64'
      )}
    >
      {/* Left side - Title */}
      <div className="flex items-center gap-4">
        {title && (
          <div>
            <h1 className="text-lg font-semibold text-foreground">{title}</h1>
            {description && (
              <p className="text-sm text-muted-foreground">{description}</p>
            )}
          </div>
        )}
      </div>

      {/* Right side - Actions */}
      <div className="flex items-center gap-3">
        {/* API Status indicator */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-secondary/50 text-xs">
          <div
            className={cn(
              'w-2 h-2 rounded-full',
              isApiHealthy ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'
            )}
          />
          <span className="text-muted-foreground">
            {isApiHealthy ? 'API Connected' : 'API Offline'}
          </span>
        </div>

        {/* Search button */}
        <Button variant="ghost" size="icon" className="h-9 w-9">
          <Search className="h-4 w-4" />
        </Button>

        {/* Notifications */}
        <Button variant="ghost" size="icon" className="h-9 w-9 relative">
          <Bell className="h-4 w-4" />
          <span className="absolute top-1 right-1 w-2 h-2 bg-primary rounded-full" />
        </Button>

        {/* Custom actions */}
        {actions}

        {/* User menu */}
        <Button variant="ghost" size="icon" className="h-9 w-9">
          <User className="h-4 w-4" />
        </Button>
      </div>
    </header>
  );
}

export default Header;
