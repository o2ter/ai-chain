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
import { Ollama, Config, ToolCall } from 'ollama';
import { ClientProvider } from '../../client/provider';
import { ChatOptions, EmbedOptions } from '../../client/types';

type OllamaEmbedConfig = Omit<Parameters<Ollama['embed']>[0], keyof EmbedOptions> & EmbedOptions;

type _OllamaChatConfig = Parameters<Ollama['chat']>[0];
type OllamaChatConfig = Omit<_OllamaChatConfig, keyof ChatOptions | 'stream'> & ChatOptions;

export class OllamaProvider extends ClientProvider {

  client: Ollama;

  constructor(options: Config) {
    super();
    this.client = new Ollama(options);
  }

  async* models() {
    const { models } = await this.client.list();
    for (const model of models) {
      yield { name: model.name };
    }
  }

  async embeddings({ signal, ...options }: OllamaEmbedConfig) {
    const {
      embeddings,
      prompt_eval_count,
    } = await this.client.embed({
      truncate: true,
      ...options
    });
    return {
      embeddings: embeddings.map(values => ({ values })),
      usage: {
        prompt_tokens: prompt_eval_count,
        total_tokens: prompt_eval_count,
      },
    };
  }

  #createChatParams({
    model,
    systemMessage,
    messages,
    tools,
    ...options
  }: OllamaChatConfig): Omit<_OllamaChatConfig, 'stream'> {
    return {
      model,
      messages: _.compact([
        systemMessage && { role: 'system', content: systemMessage },
        ...messages.map(msg => {
          const { role, content } = msg;
          switch (role) {
            case 'user':
              return { role, content };
            case 'assistant':
              return {
                role,
                content,
                thinking: msg.reasoning,
                tool_calls: msg.tool_calls?.map(call => ({
                  function: {
                    name: call.name,
                    arguments: call.arguments,
                  },
                })),
              };
            case 'tool':
              return {
                role,
                content,
                tool_name: msg.tool_call_id,
              };
          }
        }),
      ]),
      tools: tools ? _.map(tools, tool => ({
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
      })) : undefined,
      ...options,
    };
  }

  async* chatStream({ signal, ...options }: OllamaChatConfig) {

    const response = await this.client.chat({
      ...this.#createChatParams(options),
      stream: true,
    });

    const now = Date.now();
    const toolCallIds = new Map<number, string>();

    for await (const { message: { content, thinking, tool_calls }, ...data } of response) {
      if (content) yield { type: 'content', content } as const;
      if (thinking) yield { type: 'reasoning', reasoning: thinking } as const;
      if (tool_calls) {
        for (const [index, { id, function: call }] of (tool_calls as (ToolCall & { id?: string })[]).entries()) {
          if (!toolCallIds.has(index)) toolCallIds.set(index, id ?? `tool-${now}-${index}`);
          yield {
            type: 'tool_call',
            tool_call_id: toolCallIds.get(index)!,
            name: call.name,
            arguments: call.arguments as any,
          } as const;
        }
      }
      if (!_.isNil(data.prompt_eval_count) || !_.isNil(data.eval_count)) {
        yield {
          type: 'usage',
          usage: {
            prompt_tokens: data.prompt_eval_count,
            completion_tokens: data.eval_count,
            total_tokens: (data.prompt_eval_count ?? 0) + (data.eval_count ?? 0),
          },
        } as const;
      }
    }
  }
};