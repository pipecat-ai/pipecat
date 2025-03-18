interface WaveformProps {
  speakingState: 'user' | 'assistant' | 'system' | 'idle';
}

export function Waveform({ speakingState }: WaveformProps) {
  const getAnimationClass = () => {
    switch (speakingState) {
      case 'user':
        return 'waveform-bar bg-zinc-800';
      case 'assistant':
        return 'waveform-bar bg-zinc-600';
      case 'system':
        return 'waveform-bar-listening bg-zinc-400';
      default:
        return 'waveform-bar-listening bg-zinc-300';
    }
  };

  return (
    <div className="flex items-end gap-1 h-8">
      {[...Array(5)].map((_, i) => (
        <div
          key={i}
          className={`w-1 rounded-full ${getAnimationClass()}`}
          style={{
            height: speakingState === 'idle' ? '40%' : `${Math.random() * 60 + 40}%`,
            animationDelay: `${i * 0.1}s`
          }}
        />
      ))}
    </div>
  );
} 