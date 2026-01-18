'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { useAppStore } from '@/stores';
import {
  LayoutDashboard,
  PenTool,
  FileText,
  Globe,
  Presentation,
  Trophy,
  Settings,
  ChevronLeft,
  ChevronRight,
  Zap,
} from 'lucide-react';
import { Button } from '@/components/ui/button';

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Content Studio', href: '/studio', icon: PenTool },
  { name: 'Briefs', href: '/briefs', icon: FileText },
  { name: 'GEO Intelligence', href: '/geo', icon: Globe },
  { name: 'Slideshow', href: '/slides', icon: Presentation },
  { name: 'Results', href: '/results', icon: Trophy },
];

const bottomNav = [
  { name: 'Settings', href: '/settings', icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const { sidebarCollapsed, toggleSidebar } = useAppStore();

  return (
    <aside
      className={cn(
        'fixed left-0 top-0 z-40 h-screen bg-card/80 backdrop-blur-xl border-r border-border/50 transition-all duration-300 flex flex-col',
        sidebarCollapsed ? 'w-16' : 'w-64'
      )}
    >
      {/* Logo */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-border/50">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center">
            <Zap className="w-5 h-5 text-white" />
          </div>
          {!sidebarCollapsed && (
            <span className="font-bold text-lg bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
              STORM
            </span>
          )}
        </Link>
        <Button
          variant="ghost"
          size="icon"
          onClick={toggleSidebar}
          className="h-8 w-8"
        >
          {sidebarCollapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Main nav */}
      <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
        {navigation.map((item) => {
          const isActive = pathname === item.href ||
            (item.href !== '/' && pathname.startsWith(item.href));

          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200',
                isActive
                  ? 'bg-primary/10 text-primary shadow-sm'
                  : 'text-muted-foreground hover:bg-secondary hover:text-foreground',
                sidebarCollapsed && 'justify-center px-2'
              )}
              title={sidebarCollapsed ? item.name : undefined}
            >
              <item.icon className={cn('h-5 w-5 flex-shrink-0', isActive && 'text-primary')} />
              {!sidebarCollapsed && <span>{item.name}</span>}
            </Link>
          );
        })}
      </nav>

      {/* Bottom nav */}
      <div className="px-2 py-4 border-t border-border/50">
        {bottomNav.map((item) => {
          const isActive = pathname === item.href;

          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200',
                isActive
                  ? 'bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:bg-secondary hover:text-foreground',
                sidebarCollapsed && 'justify-center px-2'
              )}
              title={sidebarCollapsed ? item.name : undefined}
            >
              <item.icon className="h-5 w-5 flex-shrink-0" />
              {!sidebarCollapsed && <span>{item.name}</span>}
            </Link>
          );
        })}
      </div>
    </aside>
  );
}

export default Sidebar;
