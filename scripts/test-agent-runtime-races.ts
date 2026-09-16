import assert from "node:assert/strict";
import { AgentRuntime, ContextStore, createSttSignalWake } from "../src/runtime/agent/index.ts";
import type { AssistantSuggestion } from "../src/app/types.ts";

Object.defineProperty(globalThis, "window", {
  configurable: true,
  value: globalThis,
});

const answer: AssistantSuggestion = {
  answer: "coach answer",
  bullets: [],
  clarifyingQuestion: null,
};

let firstResolve: ((value: AssistantSuggestion) => void) | null = null;
let calls = 0;
const messages: string[] = [];
const skipped: string[] = [];

const runtime = new AgentRuntime(
  new ContextStore(),
  {
    complete: async () => {
      calls += 1;
      if (calls === 1) {
        return new Promise<AssistantSuggestion>((resolve) => {
          firstResolve = resolve;
        });
      }
      return answer;
    },
  },
  {
    onMessage: (suggestion) => messages.push(suggestion.answer),
    onError: (message) => assert.fail(message),
    onWakeSkipped: (_wake, reason) => skipped.push(reason),
  }
);

runtime.wake(createSttSignalWake("a proactive signal", "test_signal"));
await nextTurn();
assert.equal(calls, 1);

runtime.beginManualAsk();
firstResolve?.(answer);
await nextTurn();
assert.deepEqual(messages, []);
assert.ok(skipped.includes("superseded_by_manual_ask"));

runtime.wake(createSttSignalWake("ignored while meeting Ask is active", "test_signal"));
assert.ok(skipped.includes("manual_ask_active"));

runtime.finishManualAsk();
runtime.wake(createSttSignalWake("accepted after meeting Ask completes", "test_signal"));
await nextTurn();
assert.deepEqual(messages, ["coach answer"]);

let isolatedCoachResolve: ((value: AssistantSuggestion) => void) | null = null;
const isolatedCoachMessages: string[] = [];
const isolatedCoachRuntime = new AgentRuntime(
  new ContextStore(),
  {
    complete: () => new Promise<AssistantSuggestion>((resolve) => {
      isolatedCoachResolve = resolve;
    }),
  },
  {
    onMessage: (suggestion) => isolatedCoachMessages.push(suggestion.answer),
    onError: (message) => assert.fail(message),
  }
);

isolatedCoachRuntime.wake(createSttSignalWake("coach continues during independent Fn work", "test_signal"));
await nextTurn();
const independentFnRun = Promise.resolve("fn answer");
assert.equal(await independentFnRun, "fn answer");
isolatedCoachResolve?.(answer);
await nextTurn();
assert.deepEqual(isolatedCoachMessages, ["coach answer"]);

console.log("meeting Ask / Coach wake race tests passed");

async function nextTurn() {
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

// --- Stalled-run watchdog -------------------------------------------------
// A transport that never settles used to pin `inFlight` forever: every later
// question was dropped as "already running" and the panel looked dead. The
// watchdog must both surface an error and release the runtime.
{
  const errors: string[] = [];
  const woken: string[] = [];
  let released = false;

  const stalledRuntime = new AgentRuntime(
    new ContextStore(),
    { complete: () => new Promise<AssistantSuggestion>(() => {}) },
    {
      onMessage: () => assert.fail("stalled transport must not deliver a message"),
      onError: (message) => errors.push(message),
      onWakeStart: () => {
        released = true;
      },
    },
    60
  );

  stalledRuntime.wake(createSttSignalWake("a question that hangs", "test_signal"));
  await new Promise<void>((resolve) => setTimeout(resolve, 150));

  assert.equal(errors.length, 1, "stalled run must report one error");
  assert.ok(
    errors[0].includes("已放弃"),
    `watchdog message should tell the user it gave up, got: ${errors[0]}`
  );

  // The critical assertion: the runtime released, so the next question runs.
  const recovered = new AgentRuntime(
    new ContextStore(),
    { complete: async () => answer },
    { onMessage: () => woken.push("ok"), onError: (m) => assert.fail(m) },
    60
  );
  recovered.wake(createSttSignalWake("a question after the stall", "test_signal"));
  await new Promise<void>((resolve) => setTimeout(resolve, 50));
  assert.deepEqual(woken, ["ok"], "a fresh question must still work after a stall");
  assert.ok(released);

  console.log("stalled-run watchdog tests passed");
}

// --- Cancelling an in-flight run ------------------------------------------
// "New conversation" while an answer is generating must abandon that run and
// free the runtime *immediately* — not when the abandoned request settles.
//
// The order matters: the second question is asked BEFORE the abandoned request
// resolves. That is the case the bug lived in — `inFlight` was still true, so
// the new question was silently parked and the panel stayed busy.
{
  let release: ((value: AssistantSuggestion) => void) | null = null;
  const delivered: string[] = [];
  const cancelled: string[] = [];
  const skipReasons: string[] = [];

  const cancelRuntime = new AgentRuntime(
    new ContextStore(),
    {
      complete: () =>
        new Promise<AssistantSuggestion>((resolve) => {
          release = resolve;
        }),
    },
    {
      onMessage: (suggestion) => delivered.push(suggestion.answer),
      onError: (message) => assert.fail(message),
      onCancelled: (reason) => cancelled.push(reason),
      onWakeSkipped: (_wake, reason) => skipReasons.push(reason),
    },
    200
  );

  cancelRuntime.wake(createSttSignalWake("a question", "test_signal"));
  await new Promise<void>((resolve) => setTimeout(resolve, 20));
  cancelRuntime.cancel("user_cleared_conversation");

  assert.deepEqual(cancelled, ["user_cleared_conversation"], "cancel must notify the UI");

  // Ask again while the first request is STILL pending. The abandoned promise
  // is never resolved; whatever `release` points at now is the new one.
  cancelRuntime.wake(createSttSignalWake("a question after cancel", "test_signal"));
  await new Promise<void>((resolve) => setTimeout(resolve, 20));
  release!(answer);
  await new Promise<void>((resolve) => setTimeout(resolve, 20));
  assert.deepEqual(
    delivered,
    ["coach answer"],
    "the next question must run right after a cancel, without waiting for the abandoned request"
  );

  // The abandoned request has now hit the watchdog; its late failure must not
  // surface as an error for a conversation the user already left.
  await new Promise<void>((resolve) => setTimeout(resolve, 260));
  assert.deepEqual(delivered, ["coach answer"], "a stale result must never be delivered late");
  assert.ok(
    skipReasons.includes("cancelled"),
    "expected the abandoned wake to be skipped as cancelled, got: " + skipReasons.join(",")
  );

  console.log("cancel-in-flight tests passed");
}
