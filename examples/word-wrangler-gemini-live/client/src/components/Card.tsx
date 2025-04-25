export function Card({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`bg-white rounded-3xl relative card-border text-black ${className}`}
    >
      {children}
    </div>
  );
}

export function CardInner({ children }: { children: React.ReactNode }) {
  return <div className="p-6 lg:p-10 relative z-2">{children}</div>;
}
