import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { PanelLeft } from "lucide-react"
import { cn } from "./utils"


type SidebarContext = {
  state: "expanded" | "collapsed"
  open: boolean
  setOpen: (open: boolean) => void
  toggleSidebar: () => void
}

const SidebarContext = React.createContext<SidebarContext | null>(null)

function useSidebar() {
  const ctx = React.useContext(SidebarContext)
  if (!ctx) throw new Error("useSidebar must be used within SidebarProvider")
  return ctx
}

const SidebarProvider = React.forwardRef<
  HTMLDivElement,
  React.ComponentProps<"div"> & { defaultOpen?: boolean; open?: boolean; onOpenChange?: (open: boolean) => void }
>(({ defaultOpen = true, open: openProp, onOpenChange: setOpenProp, className, children, ...props }, ref) => {
  const [_open, _setOpen] = React.useState(defaultOpen)
  const open = openProp ?? _open
  const setOpen = React.useCallback((value: boolean | ((v: boolean) => boolean)) => {
    const newOpen = typeof value === "function" ? value(open) : value
    _setOpen(newOpen)
    setOpenProp?.(newOpen)
  }, [open, setOpenProp])
  const toggleSidebar = React.useCallback(() => setOpen((prev) => !prev), [setOpen])
  const state = open ? "expanded" as const : "collapsed" as const

  return (
    <SidebarContext.Provider value={{ state, open, setOpen, toggleSidebar }}>
      <div ref={ref} className={cn("flex min-h-0 flex-1", className)} {...props}>
        {children}
      </div>
    </SidebarContext.Provider>
  )
})
SidebarProvider.displayName = "SidebarProvider"

const Sidebar = React.forwardRef<HTMLDivElement, React.ComponentProps<"div">>(
  ({ className, children, ...props }, ref) => {
    const { state } = useSidebar()
    return (
      <div ref={ref} data-state={state}
        className={cn("flex flex-col border-r border-border bg-card text-foreground transition-all duration-200",
          state === "expanded" ? "w-64" : "w-12", className)}
        {...props}>
        {children}
      </div>
    )
  }
)
Sidebar.displayName = "Sidebar"

const SidebarTrigger = React.forwardRef<React.ElementRef<"button">, React.ComponentProps<"button">>(
  ({ className, ...props }, ref) => {
    const { toggleSidebar } = useSidebar()
    return (
      <button ref={ref} onClick={toggleSidebar}
        className={cn("flex h-7 w-7 items-center justify-center rounded-sm text-muted-foreground hover:text-accent hover:bg-accent/10 transition-colors", className)}
        {...props}>
        <PanelLeft size={16} />
        <span className="sr-only">Toggle Sidebar</span>
      </button>
    )
  }
)
SidebarTrigger.displayName = "SidebarTrigger"

const SidebarHeader = React.forwardRef<HTMLDivElement, React.ComponentProps<"div">>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("flex items-center gap-2 px-3 py-3 border-b border-border", className)} {...props} />
  )
)
SidebarHeader.displayName = "SidebarHeader"

const SidebarContent = React.forwardRef<HTMLDivElement, React.ComponentProps<"div">>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("flex-1 overflow-y-auto py-2", className)} {...props} />
  )
)
SidebarContent.displayName = "SidebarContent"

const SidebarGroup = React.forwardRef<HTMLDivElement, React.ComponentProps<"div">>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("px-2 py-1", className)} {...props} />
  )
)
SidebarGroup.displayName = "SidebarGroup"

const SidebarGroupLabel = React.forwardRef<HTMLDivElement, React.ComponentProps<"div">>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("px-2 py-1.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground", className)} {...props} />
  )
)
SidebarGroupLabel.displayName = "SidebarGroupLabel"

const SidebarMenu = React.forwardRef<HTMLDivElement, React.ComponentProps<"div">>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("flex flex-col gap-0.5", className)} {...props} />
  )
)
SidebarMenu.displayName = "SidebarMenu"

const SidebarMenuItem = React.forwardRef<HTMLDivElement, React.ComponentProps<"div">>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("", className)} {...props} />
  )
)
SidebarMenuItem.displayName = "SidebarMenuItem"

const SidebarMenuButton = React.forwardRef<
  HTMLButtonElement,
  React.ComponentProps<"button"> & { asChild?: boolean; isActive?: boolean }
>(({ asChild = false, isActive, className, ...props }, ref) => {
  const Comp = asChild ? Slot : "button"
  const { state } = useSidebar()
  return (
    <Comp ref={ref}
      className={cn(
        "flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm transition-colors",
        "hover:bg-accent/10 hover:text-accent",
        isActive && "bg-accent/10 text-accent",
        state === "collapsed" && "justify-center px-0",
        className)}
      {...props} />
  )
})
SidebarMenuButton.displayName = "SidebarMenuButton"

export {
  SidebarProvider, Sidebar, SidebarTrigger, SidebarHeader,
  SidebarContent, SidebarGroup, SidebarGroupLabel,
  SidebarMenu, SidebarMenuItem, SidebarMenuButton, useSidebar,
}
