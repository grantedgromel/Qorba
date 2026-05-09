import createClient from "openapi-fetch";
import type { paths } from "@qorba/shared/openapi";

const baseUrl =
  typeof window === "undefined"
    ? (process.env.API_URL ?? "http://localhost:8000")
    : "";

export const api = createClient<paths>({
  baseUrl,
  credentials: "include",
});
