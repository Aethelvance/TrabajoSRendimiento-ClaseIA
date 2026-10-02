import Lenis from "lenis";
import { animate, motion, useMotionValue, useReducedMotion, useTransform } from "motion/react";
import {
  Activity,
  Dumbbell,
  Gauge as GaugeIcon,
  HeartPulse,
  History,
  Moon,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import { useEffect, useState } from "react";
import { health, metrics, predict } from "./api.js";

const EASE = [0.16, 1, 0.3, 1];
const MIN_Y = 50;
const MAX_Y = 285;

function tier(value) {
  if (value == null) return null;
  if (value < 110) return { label: "Día suave", hint: "Tu cuerpo pide descanso: baja el ritmo hoy." };
  if (value < 160) return { label: "Nivel estable", hint: "Buen equilibrio para una jornada normal." };
  if (value < 210) return { label: "Buen nivel", hint: "Estás en un gran momento para exigirte." };
  return { label: "Nivel máximo", hint: "Día ideal para tu mejor marca." };
}

function SliderRow({ icon: Icon, label, helper, value, min, max, step, unit, onChange }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <label className="flex items-center gap-2 text-sm font-medium text-bone-100">
          <Icon size={17} strokeWidth={2} className="text-lima-300" />
          {label}
        </label>
        <span className="rounded-full bg-night-800 px-3 py-1 font-mono text-sm text-lima-300">
          {value.toFixed(1)} {unit}
        </span>
      </div>
      <input
        type="range"
        className="slider"
        style={{ "--fill": `${pct}%` }}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-label={label}
      />
      <p className="text-xs text-bone-500">{helper}</p>
    </div>
  );
}

function Gauge({ value, loading, reduce }) {
  const R = 80;
  const ARC = Math.PI * R;
  const frac = value == null ? 0 : Math.min(1, Math.max(0, (value - MIN_Y) / (MAX_Y - MIN_Y)));

  const mv = useMotionValue(0);
  const shown = useTransform(mv, (v) =>
    value == null ? "—" : String(Math.round(MIN_Y + v * (MAX_Y - MIN_Y))),
  );
  useEffect(() => {
    if (value == null || reduce) return;
    const controls = animate(mv, frac, { type: "spring", stiffness: 60, damping: 16 });
    return controls.stop;
  }, [frac, value, reduce, mv]);

  return (
    <div className="relative mx-auto w-full max-w-75">
      <svg viewBox="0 0 200 118" className="w-full" role="img" aria-label="Medidor de rendimiento">
        <path d="M 20 105 A 80 80 0 0 1 180 105" fill="none" stroke="#2a3a30" strokeWidth="14" strokeLinecap="round" />
        {loading ? (
          <path d="M 20 105 A 80 80 0 0 1 180 105" fill="none" stroke="#a3e635" strokeWidth="14" strokeLinecap="round" opacity="0.35">
            <animate attributeName="opacity" values="0.15;0.5;0.15" dur="1.2s" repeatCount="indefinite" />
          </path>
        ) : (
          <motion.path
            d="M 20 105 A 80 80 0 0 1 180 105"
            fill="none"
            stroke="#a3e635"
            strokeWidth="14"
            strokeLinecap="round"
            strokeDasharray={ARC}
            initial={false}
            animate={{ strokeDashoffset: ARC * (1 - frac) }}
            transition={
              reduce ? { duration: 0 } : { type: "spring", stiffness: 60, damping: 16 }
            }
          />
        )}
      </svg>
      <div className="absolute inset-x-0 bottom-0 text-center">
        {loading ? (
          <div className="mx-auto h-10 w-24 animate-pulse rounded-lg bg-night-700" />
        ) : (
          <motion.span className="font-mono text-5xl font-semibold text-bone-100">{shown}</motion.span>
        )}
        <p className="mt-1 text-xs tracking-[0.18em] text-bone-500 uppercase">puntos</p>
      </div>
    </div>
  );
}

export default function App() {
  const reduce = useReducedMotion();
  const [deporte, setDeporte] = useState(2.5);
  const [sueno, setSueno] = useState(6.0);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [online, setOnline] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    if (reduce) return;
    const lenis = new Lenis({ smoothWheel: true });
    let raf = requestAnimationFrame(function loop(t) {
      lenis.raf(t);
      raf = requestAnimationFrame(loop);
    });
    return () => {
      cancelAnimationFrame(raf);
      lenis.destroy();
    };
  }, [reduce]);

  useEffect(() => {
    health()
      .then((h) => setOnline(h.model_loaded))
      .catch(() => setOnline(false));
    metrics()
      .then(setModelInfo)
      .catch(() => {});
  }, []);

  async function onPredict() {
    setLoading(true);
    setError(null);
    try {
      const data = await predict(deporte, sueno);
      setResult(data.rendimiento);
      setHistory((h) =>
        [{ deporte, sueno, rendimiento: data.rendimiento, at: new Date() }, ...h].slice(0, 6),
      );
    } catch (e) {
      if (e.status === 422) setError("Revisa las horas: deporte 0–5 h y sueño 3–9 h.");
      else if (e.status === 503) setError("El modelo no está cargado en el servidor.");
      else setError("No se pudo contactar el servicio. ¿Está la API en marcha?");
    } finally {
      setLoading(false);
    }
  }

  const t = tier(result);

  return (
    <div className="min-h-dvh bg-night-950 text-bone-100">
      <header className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <HeartPulse size={20} strokeWidth={2} className="text-lima-400" />
          <span className="text-sm font-semibold tracking-tight">Pulso Diario</span>
        </div>
        <div
          className="flex items-center gap-2 rounded-full border border-night-700 px-3 py-1 text-xs text-bone-300"
          title={online == null ? "Comprobando…" : online ? "Modelo en línea" : "Sin conexión"}
        >
          <span
            className={`inline-block size-2 rounded-full ${
              online == null ? "bg-bone-500" : online ? "bg-lima-400" : "bg-red-400"
            }`}
          />
          {online == null ? "Conectando…" : online ? "Servicio en línea" : "Sin conexión"}
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 pt-10 pb-20 md:pt-14">
        <motion.div
          initial={reduce ? false : { opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: EASE }}
          className="max-w-2xl"
        >
          <h1 className="text-4xl leading-none font-semibold tracking-tighter md:text-5xl">
            ¿Cómo rendirás <span className="text-lima-300 italic">hoy?</span>
          </h1>
          <p className="mt-3 max-w-[52ch] text-base leading-relaxed text-bone-300">
            Mueve los controles con tus horas de deporte y sueño, y obtén tu puntuación al instante.
          </p>
        </motion.div>

        <div className="mt-10 grid grid-cols-1 gap-6 lg:grid-cols-5">
          <motion.section
            initial={reduce ? false : { opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.6, ease: EASE }}
            className="flex flex-col gap-7 rounded-2xl border border-night-700 bg-night-900 p-6 lg:col-span-2"
            aria-label="Tus datos"
          >
            <SliderRow
              icon={Dumbbell}
              label="Deporte"
              helper="Horas de actividad física ayer, de 0 a 5."
              value={deporte}
              min={0}
              max={5}
              step={0.1}
              unit="h"
              onChange={setDeporte}
            />
            <SliderRow
              icon={Moon}
              label="Sueño"
              helper="Horas dormidas anoche, de 3 a 9."
              value={sueno}
              min={3}
              max={9}
              step={0.1}
              unit="h"
              onChange={setSueno}
            />
            <div className="flex flex-col gap-2">
              <motion.button
                whileTap={reduce ? undefined : { scale: 0.98, y: 1 }}
                onClick={onPredict}
                disabled={loading}
                className="flex items-center justify-center gap-2 rounded-full bg-lima-400 px-6 py-3 text-sm font-semibold text-night-950 transition disabled:opacity-60"
              >
                <Sparkles size={16} strokeWidth={2.2} />
                {loading ? "Calculando…" : "Calcular mi rendimiento"}
              </motion.button>
              {error && (
                <p role="alert" className="text-sm text-red-300">
                  {error}
                </p>
              )}
            </div>
          </motion.section>

          <motion.section
            initial={reduce ? false : { opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.3 }}
            transition={{ duration: 0.6, delay: 0.08, ease: EASE }}
            className="flex flex-col items-center justify-center gap-4 rounded-2xl border border-night-700 bg-night-900 p-6 lg:col-span-3"
            aria-label="Tu resultado" aria-live="polite"
          >
            <div className="flex items-center gap-2 text-sm text-bone-300">
              <GaugeIcon size={16} strokeWidth={2} className="text-lima-300" />
              Tu puntuación
            </div>
            <Gauge value={result} loading={loading} reduce={reduce} />
            {t && !loading ? (
              <motion.div
                key={result}
                initial={reduce ? false : { opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, ease: EASE }}
                className="text-center"
              >
                <p className="text-xl font-semibold text-lima-300">{t.label}</p>
                <p className="mx-auto mt-1 max-w-[42ch] text-sm text-bone-300">{t.hint}</p>
              </motion.div>
            ) : (
              !loading && (
                <p className="max-w-[42ch] text-center text-sm text-bone-500">
                  Ajusta los controles y pulsa calcular para ver tu resultado.
                </p>
              )
            )}
          </motion.section>
        </div>

        <motion.section
          initial={reduce ? false : { opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.2 }}
          transition={{ duration: 0.6, ease: EASE }}
          className="mt-6 rounded-2xl border border-night-700 bg-night-900 p-6"
          aria-label="Tus últimas consultas"
        >
          <div className="flex items-center justify-between">
            <h2 className="flex items-center gap-2 text-sm font-semibold">
              <History size={16} strokeWidth={2} className="text-lima-300" />
              Tus últimas consultas
            </h2>
            {history.length > 0 && (
              <button
                onClick={() => setHistory([])}
                className="flex items-center gap-1 rounded-full border border-night-700 px-3 py-1 text-xs text-bone-300 transition active:translate-y-px"
              >
                <RotateCcw size={13} strokeWidth={2} />
                Limpiar
              </button>
            )}
          </div>
          {history.length === 0 ? (
            <p className="mt-3 text-sm text-bone-500">
              Aún no hay consultas. Tus resultados aparecerán aquí.
            </p>
          ) : (
            <ul className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {history.map((h, i) => (
                <motion.li
                  key={`${h.at.getTime()}-${i}`}
                  initial={reduce ? false : { opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.35, ease: EASE }}
                  className="flex items-center justify-between rounded-2xl bg-night-800 px-4 py-3"
                >
                  <span className="flex items-center gap-2 text-xs text-bone-300">
                    <Activity size={14} strokeWidth={2} className="text-lima-300" />
                    {h.deporte.toFixed(1)} h · {h.sueno.toFixed(1)} h
                  </span>
                  <span className="font-mono text-lg font-semibold text-lima-300">
                    {Math.round(h.rendimiento)}
                  </span>
                </motion.li>
              ))}
            </ul>
          )}
        </motion.section>

        <footer className="mt-10 flex flex-col gap-1 border-t border-night-700 pt-5 text-xs text-bone-500">
          <p>Pulso Diario · estimación orientativa, no es consejo médico.</p>
          {modelInfo && (
            <p>
              Precisión del modelo: {(modelInfo.r2 * 100).toFixed(1)} % en datos de prueba · error
              medio {modelInfo.mae.toFixed(1)} puntos.
            </p>
          )}
        </footer>
      </main>
    </div>
  );
}
