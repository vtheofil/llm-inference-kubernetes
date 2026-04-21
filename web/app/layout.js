import "./globals.css";

export const metadata = {
  title: "Thesis Chat",
  description: "Mini ChatGPT-like UI for thesis",
};

export default function RootLayout({ children }) {
  return (
    <html lang="el">
      <body>{children}</body>
    </html>
  );
}