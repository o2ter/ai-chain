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

type OllamaChatConfig = Parameters<Ollama['chat']>[0];

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

  async embeddings(options: EmbedOptions) {
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

  chat<S extends boolean = false>({
    model,
    messages,
    tools,
    stream,
    ...options
  }: Omit<OllamaChatConfig, keyof ChatOptions<S>> & ChatOptions<S>) {

    const params: Omit<OllamaChatConfig, 'stream'> = {
      model,
      messages: messages.map(msg => {
        const { role, content } = msg;
        switch (role) {
          case 'system':
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
          arguments: any;
        }[] = [];

        for await (const { message: { content, thinking, tool_calls }, ...data } of response) {
          if (content) yield { content: content };
          if (thinking) yield { reasoning: thinking };
          if (tool_calls) {
            for (const [index, { function: call }] of tool_calls.entries()) {
              calls[index] = {
                name: call?.name ?? calls[index]?.name ?? '',
                arguments: call?.arguments ?? calls[index]?.arguments ?? {},
              };
            }
          }
          if (!_.isNil(data.prompt_eval_count) || !_.isNil(data.eval_count)) {
            usage = {
              prompt_tokens: data.prompt_eval_count ?? 0,
              completion_tokens: data.eval_count ?? 0,
              total_tokens: (data.prompt_eval_count ?? 0) + (data.eval_count ?? 0),
            };
          }
        }
        if (!_.isEmpty(calls)) {
          yield {
            tool_calls: calls.map(call => ({
              id: call.name,
              name: call.name,
              arguments: call.arguments,
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
            id: call.function.name,
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