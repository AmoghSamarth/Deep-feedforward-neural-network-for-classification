import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Deep Feedforward Neural Network | Iris Classification",
  description: "Academic pattern recognition project for multi-class Iris classification using a Deep Feedforward Neural Network (515 parameters). Course TAE 1 by Amogh Samarth (USN: CM23034).",
  keywords: ["Deep Learning", "Neural Network", "Feedforward", "Pattern Recognition", "Iris Classification", "TensorFlow", "Keras"],
  authors: [{ name: "Amogh Samarth", url: "https://github.com/AmoghSamarth" }],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900 antialiased selection:bg-blue-100 selection:text-blue-900">
        <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6 sm:py-8 lg:px-8">
          {children}
        </div>
      </body>
    </html>
  );
}
