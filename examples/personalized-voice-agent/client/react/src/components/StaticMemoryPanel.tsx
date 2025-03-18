import React, { useState, useEffect, useCallback } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "./ui/scroll-area"
import { Memory, CaretRight } from "@phosphor-icons/react"
import { cn } from "@/lib/utils"
import { Button } from "./ui/button"
import { MemoryCard } from "./MemoryCard"
import { memoryRefreshTrigger } from "./DebugDisplay"

interface MemoryItem {
  id: string  // Kept for React key prop
  content: string
  createdAt: string
  categories: string[]
}

const fetchMemories = async () => {
  const response = await fetch("http://localhost:7860/memories")
  const data = await response.json()
  return data
}

const MemorySkeleton = () => (
  <div className="animate-pulse space-y-4">
    <div className="flex items-start space-x-4">
      <div className="h-10 w-10 rounded-full bg-neutral-200 dark:bg-neutral-800" />
      <div className="space-y-3 flex-1">
        <div className="h-4 w-1/4 bg-neutral-200 dark:bg-neutral-800 rounded" />
        <div className="space-y-2">
          <div className="h-4 w-full bg-neutral-200 dark:bg-neutral-800 rounded" />
          <div className="h-4 w-5/6 bg-neutral-200 dark:bg-neutral-800 rounded" />
        </div>
      </div>
    </div>
  </div>
)

export default function StaticMemoryPanel() {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [memories, setMemories] = useState<MemoryItem[]>([])
  const [isInitialLoading, setIsInitialLoading] = useState(true)
  const [localRefreshTrigger, setLocalRefreshTrigger] = useState(0)

  const fetchEffectMemories = useCallback(async () => {
    // Only set loading if this is the initial fetch (no memories)
    if (memories.length === 0) {
      setIsInitialLoading(true)
    }
    
    try {
      const memories = await fetchMemories()
      if (memories) {
        setMemories(memories as unknown as MemoryItem[])
      } else {
        setMemories([])
        console.log("No memories found")
      }
    } catch (error) {
      console.error("Error fetching memories:", error)
      setMemories([])
    } finally {
      setIsInitialLoading(false)
    }
  }, [memories.length])

  useEffect(() => {
    fetchEffectMemories()
  }, [fetchEffectMemories, localRefreshTrigger])

  useEffect(() => {
    const handleRefresh = (value: number) => {
      setLocalRefreshTrigger(value);
    };
    
    memoryRefreshTrigger.subscribers.add(handleRefresh);
    return () => {
      memoryRefreshTrigger.subscribers.delete(handleRefresh);
    };
  }, []);

  return (
    <Card className="w-full overflow-hidden border border-neutral-200 dark:border-neutral-800 bg-zinc-50 dark:bg-neutral-900">
      <CardHeader className="p-4 pb-2 flex flex-row items-center justify-between space-y-0 border-b border-neutral-200 dark:border-neutral-800">
        <div className="flex items-center gap-3">
          <div className={cn("h-8 w-8 rounded-full flex items-center justify-center bg-violet-100 dark:bg-violet-900/30")}>
            <Memory className="h-4 w-4 text-violet-500" weight="duotone" />
          </div>
          <div className="flex flex-col">
            <CardTitle className="text-base font-medium text-neutral-900 dark:text-neutral-100">
              Memories ({memories.length})
            </CardTitle>
          </div>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="h-8 w-8"
        >
          <CaretRight
            className={cn(
              "h-4 w-4 text-neutral-500 transition-transform",
              isCollapsed ? "" : "rotate-90"
            )}
          />
        </Button>
      </CardHeader>
      <div className={cn(
        "transition-all duration-300 ease-in-out",
        isCollapsed ? "h-0" : "h-[calc(100vh-218px)]"
      )}>
        {!isCollapsed && (
          <ScrollArea className="h-full">
            <CardContent className="py-4">
              <div className="space-y-6">
                {isInitialLoading && memories.length === 0 ? (
                  <>
                    <MemorySkeleton />
                    <MemorySkeleton />
                    <MemorySkeleton />
                  </>
                ) : (
                  memories.map((memory) => (
                    <MemoryCard key={memory.id} memory={memory} />
                  ))
                )}
              </div>
            </CardContent>
          </ScrollArea>
        )}
      </div>
    </Card>
  )
} 