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

// --- No cap on a running answer -------------------------------------------
// A 100s watchdog used to live here. It fired on any run that outlived its
// budget — including the long-but-healthy ones it was never meant to touch —
// so it is gone, together with the transport's 85s cap. A promise that never
// settles must now surface no error at all; the exit is the user's cancel
// (new conversation), which frees the runtime on the spot.
{
  const errors: string[] = [];
  const messages: string[] = [];
  let calls = 0;

  const stalledRuntime = new AgentRuntime(
    new ContextStore(),
    {
      complete: () => {
        calls += 1;
        // The first request never settles; anything after it answers normally.
        return calls === 1
          ? new Promise<AssistantSuggestion>(() => {})
          : Promise.resolve(answer);
      },
    },
    {
      onMessage: (suggestion) => messages.push(suggestion.answer),
      onError: (message) => errors.push(message),
    }
  );

  stalledRuntime.wake(createSttSignalWake("a question that hangs", "test_signal"));
  await new Promise<void>((resolve) => setTimeout(resolve, 150));

  assert.equal(calls, 1);
  assert.deepEqual(errors, [], "a slow run must not be failed by a clock");
  assert.deepEqual(messages, [], "a parked run must not deliver anything");

  stalledRuntime.cancel("user_cleared_conversation");
  stalledRuntime.wake(createSttSignalWake("a question after the cancel", "test_signal"));
  await new Promise<void>((resolve) => setTimeout(resolve, 50));
  assert.deepEqual(
    messages,
    ["coach answer"],
    "a new conversation must not wait for the abandoned run"
  );

  console.log("uncapped-run tests passed");
}

// --- Cancelling an in-flight run ------------------------------------------
// "New conversation" while an answer is generating must abandon that run and
// free the runtime *immediately* — not when the abandoned request settles.
//
// The order matters: the second question is asked BEFORE the abandoned request
// resolves. That is the case the bug lived in — `inFlight` was still true, so
// the new question was silently parked and the panel stayed busy.
{
  const resolvers: Array<(value: AssistantSuggestion) => void> = [];
  const delivered: string[] = [];
  const cancelled: string[] = [];
  const skipReasons: string[] = [];

  const cancelRuntime = new AgentRuntime(
    new ContextStore(),
    {
      complete: () =>
        new Promise<AssistantSuggestion>((resolve) => {
          resolvers.push(resolve);
        }),
    },
    {
      onMessage: (suggestion) => delivered.push(suggestion.answer),
      onError: (message) => assert.fail(message),
      onCancelled: (reason) => cancelled.push(reason),
      onWakeSkipped: (_wake, reason) => skipReasons.push(reason),
    }
  );

  cancelRuntime.wake(createSttSignalWake("a question", "test_signal"));
  await new Promise<void>((resolve) => setTimeout(resolve, 20));
  cancelRuntime.cancel("user_cleared_conversation");

  assert.deepEqual(cancelled, ["user_cleared_conversation"], "cancel must notify the UI");

  // Ask again while the first request is STILL pending: a parked request must
  // not hold the runtime, so this runs immediately.
  cancelRuntime.wake(createSttSignalWake("a question after cancel", "test_signal"));
  await new Promise<void>((resolve) => setTimeout(resolve, 20));
  resolvers[1](answer);
  await new Promise<void>((resolve) => setTimeout(resolve, 20));
  assert.deepEqual(
    delivered,
    ["coach answer"],
    "the next question must run right after a cancel, without waiting for the abandoned request"
  );

  // Only now does the abandoned request settle. With the watchdog gone this is
  // the case that still has to be swallowed, and it is the epoch/generation
  // guard — not a clock — that keeps it out of the conversation the user left.
  resolvers[0]({ answer: "stale answer", bullets: [], clarifyingQuestion: null });
  await new Promise<void>((resolve) => setTimeout(resolve, 20));
  assert.deepEqual(delivered, ["coach answer"], "a stale result must never be delivered late");
  assert.ok(
    skipReasons.includes("cancelled"),
    "expected the abandoned wake to be skipped as cancelled, got: " + skipReasons.join(",")
  );

  console.log("cancel-in-flight tests passed");
}
