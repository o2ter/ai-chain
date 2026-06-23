//
//  index.ts
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

import _ from 'lodash';
import OpenAI, { ClientOptions } from 'openai';
import { ClientProvider } from '../../client/provider';
import { ChatOptions, EmbedOptions } from '../../client/types';

type OpenAIEmbedConfig = Omit<Parameters<OpenAI['embeddings']['create']>[0], keyof EmbedOptions> & EmbedOptions;

type _OpenAIChatConfig = Parameters<OpenAI['chat']['completions']['create']>[0];
type OpenAIChatConfig = Omit<_OpenAIChatConfig, keyof ChatOptions | 'stream' | 'stream_options'> & ChatOptions;

export class OpenAIProvider extends ClientProvider {

  client: OpenAI;
  reasoningKey: string;

  constructor({ reasoningKey, ...options }: ClientOptions & { reasoningKey?: string }) {
    super();
    this.client = new OpenAI(options);
    this.reasoningKey = reasoningKey || 'reasoning_content';
  };

  async* models() {
    for await (const model of this.client.models.list()) {
      yield { name: model.id };
    }
  }

  async embeddings({ signal, ...options }: OpenAIEmbedConfig) {
    const { data, usage } = await this.client.embeddings.create(options, { signal });
    return {
      embeddings: data.toSorted((a, b) => a.index - b.index).map(item => ({ values: item.embedding })),
      usage,
    };
  }

  #convertMessage(message: ChatOptions['messages'][number]): _OpenAIChatConfig['messages'][number] {
    const { role, content } = message;
    switch (role) {
      case 'user':
        return { role: 'user', content };
      case 'assistant':
        return {
          role: 'assistant',
          content,
          [this.reasoningKey]: message.reasoning,
          tool_calls: message.tool_calls?.map(call => ({
            id: call.id,
            type: 'function',
            function: {
              name: call.name,
              arguments: call.arguments,
            },
          })),
        };
      case 'tool':
        return {
          role: 'tool',
          content: content as any,
          tool_call_id: message.tool_call_id,
        };
      default:
        throw new Error(`Unsupported message role: ${role}`);
    }
  }

  async* chat({
    model,
    systemMessage,
    messages,
    tools,
    signal,
    ...options
  }: OpenAIChatConfig) {

    const response = await this.client.chat.completions.create({
      ...options,
      stream: true,
      stream_options: {
        include_usage: true,
      },
      model,
      messages: _.compact([
        systemMessage && { role: 'system', content: systemMessage },
        ...messages.map(msg => this.#convertMessage(msg)),
      ]),
      tools: tools ? _.map(tools, tool => ({
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
      })) : undefined,
    }, { signal });

    const now = Date.now();
    const toolCallIds = new Map<number, string>();
    let usage;

    for await (const { choices: [{ delta } = {}] = [], usage: _usage } of response) {
      if (delta) {
        const { content, tool_calls } = delta;
        const reasoning = (delta as any)?.[this.reasoningKey];
        if (content) yield { type: 'content', content } as const;
        if (reasoning) yield { type: 'reasoning', reasoning } as const;
        if (tool_calls) {
          for (const { type, id, index, function: call } of tool_calls) {
            if (!toolCallIds.has(index)) {
              if (type !== 'function') continue;
              toolCallIds.set(index, id ?? `tool-${now}-${index}`);
            }
            yield {
              type: 'tool_call',
              tool_call_id: toolCallIds.get(index)!,
              name: call?.name,
              arguments: call?.arguments,
            } as const;
          }
        }
      }
      if (_usage) {
        usage = {
          completion_tokens: _usage.completion_tokens,
          prompt_tokens: _usage.prompt_tokens,
          total_tokens: _usage.total_tokens,
          reasoning_tokens: _usage.completion_tokens_details?.reasoning_tokens,
          cached_tokens: _usage.prompt_tokens_details?.cached_tokens,
        };
      }
    }

    if (usage) {
      yield { type: 'usage', usage } as const;
    }
  }
};