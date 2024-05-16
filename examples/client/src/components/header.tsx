import Logo from "./logo";

function Header() {
  return (
    <header className="header">
      <span className="logo-button">
        <Logo />
      </span>

      <nav>
        <a href="https://git.new/ai" target="_blank">
          GitHub
        </a>
        <a href="https://discord.gg/pipecat" target="_blank">
          Discord
        </a>
      </nav>
    </header>
  );
}

export default Header;
