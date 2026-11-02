"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Results" },
  { href: "/errors/", label: "Error browser" },
  { href: "/try/", label: "Try a sentence" },
  { href: "/method/", label: "Method" },
];

export function Nav() {
  const path = usePathname();
  const current = (href: string) => (href === "/" ? path === "/" : path.startsWith(href.replace(/\/$/, "")));
  return (
    <nav aria-label="Main">
      {LINKS.map((l) => (
        <Link key={l.href} href={l.href} aria-current={current(l.href) ? "page" : undefined}>
          {l.label}
        </Link>
      ))}
    </nav>
  );
}
