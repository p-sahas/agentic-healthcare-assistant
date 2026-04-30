import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Loader2, X } from "lucide-react";
import type { Patient, PatientUpdatePayload } from "@/types";

interface Props {
  open: boolean;
  patient: Patient | null;
  onClose: () => void;
  onSave: (payload: PatientUpdatePayload) => Promise<unknown>;
}

/**
 * Slide-over panel for editing the optional profile fields the user
 * skipped at registration: ``email`` and ``notes``. Read-only fields
 * (name, phone, dob, gender) are shown for reference but not editable —
 * those would need a clinical workflow to change.
 */
export function ProfileSheet({ open, patient, onClose, onSave }: Props) {
  const [email, setEmail] = useState("");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (!patient) return;
    setEmail(patient.email ?? "");
    setNotes(patient.notes ?? "");
    setErr(null);
    setSaved(false);
  }, [patient, open]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (busy || !patient) return;
    setBusy(true);
    setErr(null);
    try {
      await onSave({
        email: email.trim() || null,
        notes: notes.trim() || null,
      });
      setSaved(true);
      window.setTimeout(() => setSaved(false), 1800);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <AnimatePresence>
      {open && patient && (
        <>
          <motion.div
            key="backdrop"
            className="fixed inset-0 bg-black/60 z-30"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.aside
            key="panel"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 28, stiffness: 280 }}
            className="fixed top-0 right-0 bottom-0 w-full sm:w-96 z-40 bg-bg-soft border-l border-border flex flex-col"
          >
            <header className="h-14 px-4 flex items-center justify-between border-b border-border">
              <h2 className="text-sm font-semibold text-slate-100">Patient profile</h2>
              <button type="button" onClick={onClose} className="btn-ghost p-1.5">
                <X size={16} />
              </button>
            </header>

            <form onSubmit={submit} className="flex-1 overflow-y-auto p-4 space-y-4">
              {/* Read-only identity */}
              <section className="space-y-1.5 pb-4 border-b border-border">
                <RO label="Full name" value={patient.full_name} />
                <RO label="Phone" value={patient.phone} />
                <RO label="DOB" value={patient.dob ?? "—"} />
                <RO label="Gender" value={genderLabel(patient.gender)} />
                <p className="text-[10px] text-slate-500 pt-1">
                  Identity fields are managed by the clinic — contact reception to change.
                </p>
              </section>

              {/* Editable */}
              <section className="space-y-3">
                <Field label="Email">
                  <input
                    type="email"
                    className="input"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="optional"
                    disabled={busy}
                  />
                </Field>
                <Field label="Notes">
                  <textarea
                    className="input min-h-[120px]"
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    placeholder="Allergies, conditions, anything you'd like the assistant to remember"
                    disabled={busy}
                  />
                </Field>
              </section>

              {err && <div className="text-xs text-danger">{err}</div>}
              {saved && <div className="text-xs text-success">Saved.</div>}
            </form>

            <footer className="p-3 border-t border-border bg-bg-soft">
              <button
                type="button"
                onClick={submit}
                className="btn-primary w-full"
                disabled={busy}
              >
                {busy ? <Loader2 size={14} className="animate-spin" /> : null}
                {busy ? "Saving…" : "Save changes"}
              </button>
            </footer>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

function genderLabel(g: Patient["gender"]) {
  if (g === "M") return "Male";
  if (g === "F") return "Female";
  if (g === "X") return "Prefer not to say";
  return "—";
}

function RO({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-2 text-sm">
      <span className="text-slate-500">{label}</span>
      <span className="text-slate-200">{value}</span>
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
