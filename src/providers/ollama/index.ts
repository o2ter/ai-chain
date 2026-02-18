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
import { Ollama, Config } from 'ollama';
import { ClientProvider } from '../../client/provider';
import { ChatOptions, ChatResponse, EmbedOptions } from '../../client/types';

export class OllamaProvider extends ClientProvider {

  client: Ollama;

  constructor(options: Config) {
    super();
    this.client = new Ollama(options);
  }

  async models() {
    const { models } = await this.client.list();
    return models.map(model => ({
      name: model.name,
    }));
  }

  async embeddings(options: EmbedOptions) {
    const {
      embeddings,
      prompt_eval_count,
    } = await this.client.embed(options);
    return {
      embeddings,
      usage: {
        prompt_tokens: prompt_eval_count,
        total_tokens: prompt_eval_count,
      },
    };
  }

  chat<S extends boolean = false>({
    model,
    messages,
    tools,
    stream,
    ...options
  }: Omit<Parameters<Ollama['chat']>[0], keyof ChatOptions<S>> & ChatOptions<S>) {

    const params = {
      model,
      messages,
      tools: tools ? _.map(tools, tool => ({
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
      }) as const) : undefined,
      ...options,
    } as const;

    if (stream) {
      const self = this;

      return (async function* () {

        const response = await self.client.chat({
          stream: true,
          ...params,
        });

        let usage;
        const calls: {
          name: string;
          arguments: string;
        }[] = [];

        for await (const part of response) {
          const { message: { content, thinking, tool_calls }, ...rest } = part;
          if (content) yield { content: content };
          if (thinking) yield { reasoning: thinking };
          if (tool_calls) {
            for (const [index, { function: call }] of tool_calls.entries()) {
              calls[index] = {
                name: `${calls[index]?.name ?? ''}${call?.name ?? ''}`,
                arguments: `${calls[index]?.arguments ?? ''}${call?.arguments ?? ''}`,
              };
            }
          }
          if (!_.isNil(rest.prompt_eval_count) || !_.isNil(rest.eval_count)) {
            usage = {
              prompt_tokens: rest.prompt_eval_count ?? 0,
              completion_tokens: rest.eval_count ?? 0,
              total_tokens: (rest.prompt_eval_count ?? 0) + (rest.eval_count ?? 0),
            };
          }
        }
        if (!_.isEmpty(calls)) {
          yield {
            tool_calls: calls.map(call => ({
              name: call.name,
              arguments: JSON.parse(call.arguments),
            })),
          };
        }
        if (usage) yield { usage };

      })() as ChatResponse<S>;

    } else {

      return (async () => {

        const response = await this.client.chat({
          ...params,
        });

        return {
          content: response.message.content,
          reasoning: response.message.thinking,
          tool_calls: response.message.tool_calls?.map(call => ({
            name: call.function.name,
            arguments: call.function.arguments,
          })),
          usage: {
            prompt_tokens: response.prompt_eval_count,
            completion_tokens: response.eval_count,
            total_tokens: response.prompt_eval_count + response.eval_count,
          },
        };

      })() as ChatResponse<S>;
    }
  }
};