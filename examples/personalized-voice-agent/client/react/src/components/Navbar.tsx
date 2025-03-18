import { ConnectButton } from './ConnectButton'
import ThemeAwareLogo from './ThemeAwareLogo'

const Navbar = ({ onReset }: { onReset: () => void }) => {
  return (
    <nav className="fixed top-4 left-1/2 -translate-x-1/2 w-full max-w-[95%] bg-white rounded-3xl shadow-lg px-6 py-2 flex items-center justify-between z-50">
        <div onClick={onReset} className="cursor-pointer">
            <ThemeAwareLogo />
        </div>
        <button className="p-2 hover:bg-zinc-100 rounded-full transition-colors">
          <ConnectButton />
        </button>
      </nav>
  )
}

export default Navbar