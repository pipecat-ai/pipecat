import { useEffect, useRef } from "react"
import { cn } from "@/lib/utils"

interface BorderTrailProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: number
  duration?: number
}

export const BorderTrail: React.FC<BorderTrailProps> = ({ className, size = 100, duration = 5, ...props }) => {
  const borderRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const borderElement = borderRef.current
    if (!borderElement) return

    const animateBorder = () => {
      const startTime = performance.now()
      let animationFrameId: number

      const animate = (currentTime: number) => {
        const elapsed = (currentTime - startTime) / 1000
        const progress = (elapsed % duration) / duration
        const rotation = progress * 360

        if (borderElement) {
          borderElement.style.transform = `rotate(${rotation}deg)`
        }

        animationFrameId = requestAnimationFrame(animate)
      }

      animationFrameId = requestAnimationFrame(animate)

      return () => {
        cancelAnimationFrame(animationFrameId)
      }
    }

    const cleanup = animateBorder()
    return cleanup
  }, [duration])

  return (
    <div
      ref={borderRef}
      className={cn("absolute inset-0 rounded-full opacity-70", className)}
      style={{
        width: `${size}%`,
        height: `${size}%`,
        left: `${(100 - size) / 2}%`,
        top: `${(100 - size) / 2}%`,
      }}
      {...props}
    />
  )
} 