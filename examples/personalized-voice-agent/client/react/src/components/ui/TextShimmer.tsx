import { useEffect, useRef } from "react"
import { cn } from "@/lib/utils"

interface TextShimmerProps extends React.HTMLAttributes<HTMLDivElement> {
  duration?: number
  children: React.ReactNode
}

export const TextShimmer: React.FC<TextShimmerProps> = ({ className, duration = 3, children, ...props }) => {
  const shimmerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const shimmerElement = shimmerRef.current
    if (!shimmerElement) return

    const animateShimmer = () => {
      const startTime = performance.now()
      let animationFrameId: number

      const animate = (currentTime: number) => {
        const elapsed = (currentTime - startTime) / 1000
        const progress = (elapsed % duration) / duration
        const position = progress * 200 - 100 // -100% to 100%

        if (shimmerElement) {
          shimmerElement.style.backgroundPosition = `${position}% 0`
        }

        animationFrameId = requestAnimationFrame(animate)
      }

      animationFrameId = requestAnimationFrame(animate)

      return () => {
        cancelAnimationFrame(animationFrameId)
      }
    }

    const cleanup = animateShimmer()
    return cleanup
  }, [duration])

  return (
    <div
      ref={shimmerRef}
      className={cn(
        "inline-block bg-gradient-to-r from-transparent via-violet-400/20 to-transparent bg-[length:200%_100%]",
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
} 