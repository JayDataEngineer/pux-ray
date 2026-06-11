import { useState, useRef, useEffect } from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useEnhanceStore } from "@/stores/enhancement"
import { Message } from "lucide-react"
import { Loader2, Sparkles, RotateCcw } from "lucide-react"

interface ChatMessage {
  role: "user" | "assistant" | "system"
  content: string
  timestamp: number
}

interface LLMChatDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function LLMChatDialog({ open, onOpenChange }: LLMChatDialogProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  
  const activeModel = useEnhanceStore((s) => s.activeModel())

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Reset chat when dialog closes
  useEffect(() => {
    if (!open) {
      setMessages([])
      setInput("")
      setError(null)
    }
  }, [open])

  const handleSend = async () => {
    const userMessage = input.trim()
    if (!userMessage || isLoading) return

    if (!activeModel?.keyId) {
      setError("No LLM configured. Please add an LLM endpoint first.")
      return
    }

    // Add user message
    const userMsg: ChatMessage = {
      role: "user",
      content: userMessage,
      timestamp: Date.now(),
    }
    setMessages((prev) => [...prev, userMsg])
    setInput("")
    setIsLoading(true)
    setError(null)

    try {
      // Call the secure enhance endpoint
      const response = await fetch("/v1/llm/enhance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          key_id: activeModel.keyId,
          system_prompt: "You are a helpful AI assistant. Be concise and friendly.",
          prompt: userMessage,
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`LLM error (${response.status}): ${errorText.slice(0, 200)}`)
      }

      const data = await response.json()
      
      // Add assistant response
      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: data.result || "No response from model",
        timestamp: Date.now(),
      }
      setMessages((prev) => [...prev, assistantMsg])
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to get response"
      setError(errorMsg)
      setMessages((prev) => [...prev, {
        role: "system",
        content: `Error: ${errorMsg}`,
        timestamp: Date.now(),
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleStartOver = () => {
    setMessages([])
    setInput("")
    setError(null)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px] w-[90vw] h-[70vh] max-h-[70vh] flex flex-col p-0 gap-0">
        <DialogHeader className="px-4 py-3 border-b shrink-0 flex-row items-center justify-between space-y-0">
          <div className="flex items-center gap-2">
            <Message className="h-4 w-4" />
            <DialogTitle className="text-sm">AI Chat</DialogTitle>
            {activeModel && (
              <Badge variant="outline" className="text-[9px]">
                {activeModel.name}
              </Badge>
            )}
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 text-xs"
            onClick={handleStartOver}
            disabled={messages.length === 0}
          >
            <RotateCcw className="h-3 w-3 mr-1" />
            Start Over
          </Button>
        </DialogHeader>

        <DialogDescription className="sr-only">
          Chat with AI using your configured LLM endpoint
        </DialogDescription>

        <div className="flex-1 flex flex-col min-h-0">
          {/* Messages area */}
          <ScrollArea className="flex-1 px-4 py-3">
            <div ref={scrollRef} className="space-y-3">
              {messages.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full text-center py-12">
                  <Sparkles className="h-12 w-12 text-muted-foreground mb-3" />
                  <p className="text-sm text-muted-foreground mb-1">
                    Start a conversation with AI
                  </p>
                  <p className="text-xs text-muted-foreground/60">
                    {activeModel 
                      ? `Using ${activeModel.name} (${activeModel.model})`
                      : "Configure an LLM endpoint first"}
                  </p>
                </div>
              )}

              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${
                    msg.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg px-3 py-2 ${
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : msg.role === "system"
                        ? "bg-destructive/10 text-destructive text-xs"
                        : "bg-muted"
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap break-words">
                      {msg.content}
                    </p>
                    <span className="text-[9px] opacity-60 mt-1 block">
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-muted rounded-lg px-3 py-2 flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span className="text-sm text-muted-foreground">Thinking...</span>
                  </div>
                </div>
              )}

              {error && !isLoading && (
                <div className="flex justify-start">
                  <div className="bg-destructive/10 text-destructive rounded-lg px-3 py-2 text-xs max-w-[80%]">
                    {error}
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>

          {/* Input area */}
          <div className="border-t p-3 shrink-0">
            <div className="flex gap-2">
              <Input
                placeholder={activeModel ? "Type your message..." : "Configure an LLM endpoint first"}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={isLoading || !activeModel}
                className="flex-1 text-sm"
              />
              <Button
                size="sm"
                onClick={handleSend}
                disabled={!input.trim() || isLoading || !activeModel}
                className="shrink-0"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  "Send"
                )}
              </Button>
            </div>
            {!activeModel && (
              <p className="text-[10px] text-muted-foreground mt-2">
                ⚠️ No LLM configured. Add an endpoint in AI Prompt Enhancement settings.
              </p>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
