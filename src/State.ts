import { atom } from "recoil";

export const iframeURLAtom = atom({
  key: "iframeURLAtom",
  default: "http://127.0.0.1:5000",
});
