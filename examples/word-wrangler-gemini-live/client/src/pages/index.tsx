import { Card, CardInner } from "@/components/Card";
import { WordWrangler } from "@/components/Game/WordWrangler";
import { StartGameButton } from "@/components/StartButton";
import { GAME_TEXT } from "@/constants/gameConstants";
import { useConfigurationSettings } from "@/contexts/Configuration";
import { PERSONALITY_PRESETS, PersonalityType } from "@/types/personality";
import {
  IconArrowForwardUp,
  IconCheck,
  IconCode,
  IconX,
} from "@tabler/icons-react";
import JSConfetti from "js-confetti";
import Image from "next/image";
import Link from "next/link";
import { useEffect, useState } from "react";
import Logo from "../assets/logo.png";
import Star from "../assets/star.png";

export default function Home() {
  const [hasStarted, setHasStarted] = useState(false);
  const [gameEnded, setGameEnded] = useState(false);
  const [score, setScore] = useState(0);
  const [bestScore, setBestScore] = useState(0);
  const config = useConfigurationSettings();

  useEffect(() => {
    if (gameEnded) {
      const confetti = new JSConfetti();
      confetti.addConfetti({
        emojis: ["⭐", "⚡️", "👑", "✨", "💫", "🏆", "💯"],
      });
    }
  }, [gameEnded]);

  if (gameEnded) {
    return (
      <div className="flex flex-col justify-between lg:justify-center items-center min-h-[100dvh] py-4">
        <div className="flex flex-1 w-full">
          <Card className="w-full lg:max-w-2xl mx-auto mt-[50px] lg:mt-[120px] self-center text-center pt-[62px]">
            <div className="flex items-center justify-center w-[162px] h-[162px] rounded-full absolute z-20 -top-[81px] left-1/2 -translate-x-1/2 animate-bounce-in">
              <Image src={Star} alt="Star" priority />
            </div>
            <CardInner>
              <h2 className="text-xl font-extrabold">{GAME_TEXT.finalScore}</h2>
              <p className="text-4xl font-extrabold text-emerald-700 bg-emerald-50 rounded-full px-4 py-4 my-4">
                {score}
              </p>
              <p className="font-medium text-slate-500">
                {GAME_TEXT.finalScoreMessage}{" "}
                <span className="text-slate-700 font-extrabold">
                  {bestScore}
                </span>
              </p>
              <div className="h-[1px] bg-slate-200 my-6" />
              <div className="flex items-center justify-center">
                <Link
                  href="https://github.com/daily-co/word-wrangler-gemini-live"
                  className="button ghost w-full lg:w-auto"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <IconCode size={24} />
                  View project source code
                </Link>
              </div>
            </CardInner>
          </Card>
        </div>
        <footer className="flex flex-col justify-center w-full py-4 lg:py-12">
          <StartGameButton
            isGameEnded={true}
            onGameStarted={() => {
              setGameEnded(false);
              setScore(0);
              setHasStarted(true);
            }}
          />
        </footer>
      </div>
    );
  }

  if (!hasStarted) {
    return (
      <div className="flex flex-col justify-between items-center min-h-[100dvh] py-4 overflow-hidden">
        <div className="flex flex-1">
          <Card className="lg:min-w-2xl mx-auto mt-[50px] lg:mt-[120px] self-center">
            <Image
              src={Logo}
              alt="Word Wrangler"
              className="logo size-[150px] lg:size-[278px] absolute top-[-75px] lg:top-[-139px] left-[50%] -translate-x-1/2 z-10 animate-bounce-in"
              priority
            />

            <CardInner>
              <div className="flex flex-col gap-5 lg:gap-8 text-center mt-[50px] lg:mt-[100px]">
                <h2 className="text-xl font-extrabold">
                  {GAME_TEXT.introTitle}
                </h2>
                <div className="flex flex-col gap-3 lg:gap-4">
                  <div className="flex flex-row gap-3 relative">
                    <div className="absolute -top-3 -left-3 border-3 border-white lg:static size-10 lg:size-12 bg-emerald-100 text-emerald-500 rounded-full flex items-center justify-center font-semibold">
                      <IconCheck size={24} />
                    </div>
                    <div className="flex-1 flex h-[53px] lg:h-auto bg-slate-100 rounded-full text-slate-500 leading-5 px-12 items-center justify-center font-semibold text-pretty text-sm lg:text-base">
                      {GAME_TEXT.introGuide1}
                    </div>
                  </div>
                  <div className="flex flex-row gap-3 relative">
                    <div className="absolute -top-3 -left-3 border-3 border-white lg:static size-10 lg:size-12 bg-red-100 text-red-500 rounded-full flex items-center justify-center font-semibold">
                      <IconX size={24} />
                    </div>
                    <div className="flex-1 flex h-[53px] lg:h-auto bg-slate-100 rounded-full text-slate-500 leading-5 px-12 items-center justify-center font-semibold text-pretty text-sm lg:text-base">
                      {GAME_TEXT.introGuide2}
                    </div>
                  </div>
                  <div className="flex flex-row gap-3 relative">
                    <div className="absolute -top-3 -left-3 border-3 border-white lg:static size-10 lg:size-12 bg-slate-100 text-slate-400 rounded-full flex items-center justify-center font-semibold">
                      <IconArrowForwardUp size={24} />
                    </div>
                    <div className="flex-1 flex h-[53px] lg:h-auto bg-slate-100 rounded-full text-slate-500 leading-5 px-12 items-center justify-center font-semibold text-pretty text-sm lg:text-base">
                      {GAME_TEXT.introGuide3}
                    </div>
                  </div>
                </div>
              </div>
              <div className="flex-1 bg-slate-100 h-[1px] my-4 lg:my-6" />
              <div>
                <label className="font-bold flex flex-col gap-2 flex-1">
                  {GAME_TEXT.aiPersonality}
                  <select
                    className="rounded-xl h-11 font-normal"
                    value={config.personality}
                    onChange={(e) =>
                      config.setPersonality(e.target.value as PersonalityType)
                    }
                  >
                    {Object.entries(PERSONALITY_PRESETS).map(
                      ([value, label]) => (
                        <option key={value} value={value}>
                          {label}
                        </option>
                      )
                    )}
                  </select>
                </label>
              </div>
            </CardInner>
          </Card>
        </div>
        <footer className="flex flex-col justify-center w-full py-4 lg:py-12">
          <StartGameButton onGameStarted={() => setHasStarted(true)} />
        </footer>
      </div>
    );
  }

  return (
    <WordWrangler
      onGameEnded={(score, bestScore = 0) => {
        setScore(score);
        setBestScore(bestScore);
        setGameEnded(true);
      }}
    />
  );
}
