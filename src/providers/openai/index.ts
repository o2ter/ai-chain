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
import { ChatOptions, ChatResponse, EmbedOptions } from '../../client/types';

export class OpenAIProvider extends ClientProvider {

  client: OpenAI;

  constructor(options: ClientOptions) {
    super();
    this.client = new OpenAI(options);
  };

  async models() {
    const models = [];
    for await (const model of this.client.models.list()) {
      models.push({
        name: model.id,
      });
    }
    return models;
  }

  async embeddings(options: EmbedOptions) {
    const { data, usage } = await this.client.embeddings.create(options);
    return {
      embeddings: data.toSorted((a, b) => a.index - b.index).map(item => item.embedding),
      usage,
    };
  }

  chat<S extends boolean = false>({
    model,
    messages,
    tools,
    stream,
    ...options
  }: Omit<Parameters<OpenAI['chat']['completions']['create']>[0], keyof ChatOptions<S>> & ChatOptions<S>) {

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

        const response = await self.client.chat.completions.create({
          stream: true,
          stream_options: {
            include_usage: true,
          },
          ...params,
        });

        let usage;
        const calls: {
          name: string;
          arguments: string;
        }[] = [];

        for await (const part of response) {
          const { choices: [{ delta: { content, tool_calls } }] = [], usage: _usage } = part;
          if (content) yield { content };
          if (usage) usage = _usage;
          if (tool_calls) {
            for (const { type, index, function: call } of tool_calls) {
              if (type === 'function') {
                calls[index] = {
                  name: `${calls[index]?.name ?? ''}${call?.name ?? ''}`,
                  arguments: `${calls[index]?.arguments ?? ''}${call?.arguments ?? ''}`,
                };
              }
            }
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

        const response = await this.client.chat.completions.create({
          ...params,
        });

        const { choices: [{ message }] = [], usage } = response;

        return {
          content: message?.content ?? '',
          tool_calls: message?.tool_calls?.flatMap(call => call.type === 'function' ? ({
            name: call.function.name,
            arguments: JSON.parse(call.function.arguments),
          }) : []),
          usage,
        };

      })() as ChatResponse<S>;
    }
  }
};