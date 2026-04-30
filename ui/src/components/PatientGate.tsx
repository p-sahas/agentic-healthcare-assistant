import { useState, type FormEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { HeartPulse, Loader2, Phone, UserPlus, ArrowLeft } from "lucide-react";
import type { Gender, PatientRegisterPayload } from "@/types";

interface Props {
  onLogin: (phone: string) => Promise<{ ok: true } | { ok: false; status: number; message: string }>;
  onRegister: (payload: PatientRegisterPayload) => Promise<{ ok: true } | { ok: false; status: number; message: string }>;
}

type Mode = "phone" | "register";

/**
 * Full-screen identity gate. The user enters a phone number; we look it
 * up. If we find a patient → continue to chat. If we don't → slide in a
 * registration form that pre-fills the phone they typed and collects the
 * remaining required fields (full_name, dob, gender).
 */
export function PatientGate({ onLogin, onRegister }: Props) {
  const [mode, setMode] = useState<Mode>("phone");
  const [phone, setPhone] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // Registration-only fields
  const [fullName, setFullName] = useState("");
  const [dob, setDob] = useState("");
  const [gender, setGender] = useState<Gender>("F");

  const handlePhoneSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!phone.trim() || busy) return;
    setBusy(true);
    setError(null);
    const r = await onLogin(phone.trim());
    setBusy(false);
    if (r.ok) return;
    if (r.status === 404) {
      // No such patient — slide into registration
      setMode("register");
    } else {
      setError(r.message || "Lookup failed. Try again.");
    }
  };

  const handleRegisterSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (busy) return;
    if (!fullName.trim() || !phone.trim() || !dob || !gender) {
      setError("All fields are required");
      return;
    }
    setBusy(true);
    setError(null);
    const r = await onRegister({
      full_name: fullName.trim(),
      phone: phone.trim(),
      dob,
      gender,
    });
    setBusy(false);
    if (!r.ok) {
      if (r.status === 409) setError("That phone is already registered. Try logging in instead.");
      else setError(r.message || "Registration failed.");
    }
  };

  return (
    <div className="h-full flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="mx-auto size-14 rounded-2xl bg-brand-500/15 border border-brand-500/40 flex items-center justify-center mb-3">
            <HeartPulse size={26} className="text-brand-400" />
          </div>
          <h1 className="text-xl font-semibold text-slate-100">Nawaloka Health Assistant</h1>
          <p className="text-sm text-slate-400 mt-1">
            Enter your phone number to continue. New patients can register.
          </p>
        </div>

        <AnimatePresence mode="wait">
          {mode === "phone" ? (
            <motion.form
              key="phone"
              onSubmit={handlePhoneSubmit}
              initial={{ opacity: 0, x: -16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -16 }}
              transition={{ duration: 0.18 }}
              className="card p-5 space-y-4"
            >
              <div>
                <label className="text-xs uppercase tracking-wide text-slate-500">
                  <Phone size={11} className="inline -mt-0.5 mr-1" /> Phone number
                </label>
                <input
                  className="input mt-1.5"
                  value={phone}
                  onChange={(e) => setPhone(e.target.value)}
                  placeholder="078 103 0736 or +94781030736"
                  autoFocus
                  disabled={busy}
                />
                <p className="text-[10px] text-slate-500 mt-1">
                  Try <code className="text-slate-300">+94781030736</code> for the seeded demo patient.
                </p>
              </div>

              {error && <div className="text-xs text-danger">{error}</div>}

              <button type="submit" className="btn-primary w-full" disabled={busy || !phone.trim()}>
                {busy ? <Loader2 size={14} className="animate-spin" /> : null}
                {busy ? "Looking up…" : "Continue"}
              </button>

              <button
                type="button"
                onClick={() => setMode("register")}
                className="text-xs text-slate-400 hover:text-brand-400 w-full text-center"
                disabled={busy}
              >
                <UserPlus size={12} className="inline -mt-0.5 mr-1" />
                I'm new — register me
              </button>
            </motion.form>
          ) : (
            <motion.form
              key="register"
              onSubmit={handleRegisterSubmit}
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 16 }}
              transition={{ duration: 0.18 }}
              className="card p-5 space-y-3"
            >
              <button
                type="button"
                onClick={() => { setMode("phone"); setError(null); }}
                className="text-xs text-slate-400 hover:text-slate-200 inline-flex items-center gap-1 mb-1"
                disabled={busy}
              >
                <ArrowLeft size={12} /> back
              </button>

              <div className="text-sm text-slate-300 mb-1">
                We didn't find <code className="text-brand-400">{phone}</code>. Register as a new patient:
              </div>

              <Field label="Full name">
                <input
                  className="input"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="Anushka Perera"
                  required
                  disabled={busy}
                  autoFocus
                />
              </Field>

              <Field label="Phone number">
                <input
                  className="input"
                  value={phone}
                  onChange={(e) => setPhone(e.target.value)}
                  required
                  disabled={busy}
                />
              </Field>

              <div className="grid grid-cols-2 gap-3">
                <Field label="Date of birth">
                  <input
                    type="date"
                    className="input"
                    value={dob}
                    onChange={(e) => setDob(e.target.value)}
                    required
                    disabled={busy}
                  />
                </Field>
                <Field label="Gender">
                  <select
                    className="input"
                    value={gender}
                    onChange={(e) => setGender(e.target.value as Gender)}
                    required
                    disabled={busy}
                  >
                    <option value="F">Female</option>
                    <option value="M">Male</option>
                    <option value="X">Prefer not to say</option>
                  </select>
                </Field>
              </div>

              <p className="text-[10px] text-slate-500">
                Email and notes are optional — you can add them in your profile later.
              </p>

              {error && <div className="text-xs text-danger">{error}</div>}

              <button type="submit" className="btn-primary w-full" disabled={busy}>
                {busy ? <Loader2 size={14} className="animate-spin" /> : <UserPlus size={14} />}
                {busy ? "Registering…" : "Create account"}
              </button>
            </motion.form>
          )}
        </AnimatePresence>

        <p className="text-[10px] text-slate-600 text-center mt-4">
          No password. Your phone is your identity for this session.
        </p>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs uppercase tracking-wide text-slate-500">{label}</span>
      <div className="mt-1.5">{children}</div>
    </label>
  );
}
