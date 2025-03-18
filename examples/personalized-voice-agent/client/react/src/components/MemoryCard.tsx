import type React from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tag, Clock } from "lucide-react"
import { Memory } from "@phosphor-icons/react"
import { formatDistanceToNow } from "date-fns"
import { cn } from "@/lib/utils"

interface MemoryCardProps {
  memory: {
    id: string
    content: string
    createdAt: string
    tags?: string[]
    categories?: string[]
  }
}

export const MemoryCard: React.FC<MemoryCardProps> = ({ memory }) => {
  return (
    <Card className="w-full overflow-hidden border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900">
      <CardContent className="p-4">
        <div className="flex items-center gap-3 mb-4">
          <div
            className={cn(
              "h-8 w-8 rounded-full flex items-center justify-center",
              "bg-violet-100 dark:bg-violet-900/30",
            )}
          >
            <Memory className="h-4 w-4 text-violet-500" weight="duotone" />
          </div>
          <div className="flex items-center gap-2 text-xs text-neutral-500 dark:text-neutral-400">
            <Clock className="h-3 w-3" />
            <span>{formatDistanceToNow(new Date(memory.createdAt), { addSuffix: true })}</span>
          </div>
        </div>

        <div className="relative pl-4">
          <div className="absolute -left-0 top-0 bottom-0 w-[2px] rounded-full bg-violet-100 dark:bg-violet-900" />

          <div className="prose prose-sm dark:prose-invert max-w-none">
            <p className="text-sm text-neutral-800 dark:text-neutral-200 leading-relaxed">{memory.content}</p>
          </div>

          {memory.tags && memory.tags.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-3">
              {memory.tags.map((tag, tagIndex) => (
                <Badge
                  key={tagIndex}
                  variant="secondary"
                  className="px-1.5 py-0 h-4 text-[10px] bg-violet-50 dark:bg-violet-900/20 text-violet-600 dark:text-violet-400 hover:bg-violet-100 dark:hover:bg-violet-900/40 transition-colors"
                >
                  <Tag className="h-2.5 w-2.5 mr-1" />
                  {tag}
                </Badge>
              ))}
            </div>
          )}

          {memory.categories && memory.categories.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-3">
              {memory.categories.map((category, categoryIndex) => (
                <Badge
                  key={categoryIndex}
                  variant="secondary"
                  className="px-1.5 py-0 h-4 text-[10px] bg-violet-50 dark:bg-violet-900/20 text-violet-600 dark:text-violet-400 hover:bg-violet-100 dark:hover:bg-violet-900/40 transition-colors"
                >
                  <Tag className="h-2.5 w-2.5 mr-1" />
                  {category}
                </Badge>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
} 