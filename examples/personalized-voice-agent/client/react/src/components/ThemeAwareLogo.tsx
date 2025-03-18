
import mem0Logo from "./light.svg";

export default function ThemeAwareLogo({
  width = 100,
  height = 35,
}: {
  width?: number;
  height?: number;
}) {
  return <img src={mem0Logo} alt="Mem0.ai" width={width} height={height} />;
}
