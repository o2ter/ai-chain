//
//  types.ts
//
//  The MIT License
//  Copyright (c) 2021 - 2026 O2ter Limited. All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

export type ToolCall = {
  id: string;
  name: string;
  arguments: string;
};

export type EmbedOptions = {
  model: string;
  input: string | string[];
  dimensions?: number;
  signal?: AbortSignal;
};

export type EmbedResponse = {
  embeddings: {
    values: number[];
    truncated?: boolean;
  }[];
  usage?: {
    prompt_tokens: number;
    total_tokens: number;
  };
};

export type ContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } };

export type ChatUserMessage = {
  role: 'user';
  content: string | ContentPart[];
};
export type ChatAssistantMessage = {
  role: 'assistant';
  content: string;
  reasoning?: string;
  tool_calls?: ToolCall[];
};
export type ChatToolMessage = {
  role: 'tool';
  content: string;
  tool_call_id: string;
  name: string;
};

export type ChatMessage =
  | ChatUserMessage
  | ChatAssistantMessage
  | ChatToolMessage;

export type ChatTool = {
  name: string;
  description: string;
  parameters?: any;
};

export type ChatOptions = {
  model: string;
  systemMessage?: string;
  messages: ChatMessage[];
  tools?: ChatTool[];
  signal?: AbortSignal;
};

export type ChatResponseUsage = {
  completion_tokens?: number;
  prompt_tokens?: number;
  total_tokens?: number;
  reasoning_tokens?: number;
  cached_tokens?: number;
};

export type ChatResponseChunk =
  | { type: 'content'; content: string }
  | { type: 'reasoning'; reasoning: string }
  | { type: 'tool_call'; tool_call_id: string; name?: string; arguments?: string }
  | { type: 'usage'; usage: ChatResponseUsage };