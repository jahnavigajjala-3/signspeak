import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'SignSpeak - Continuous Sign Language Translation',
  description: 'AI/ML continuous sign-language understanding powered by CNN-LSTM Attention model.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-surface-base text-slate-100 antialiased min-h-screen flex flex-col">
        {children}
      </body>
    </html>
  );
}
