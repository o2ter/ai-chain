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
import { ChatOptions, ContentPart, EmbedOptions } from '../../client/types';

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

  async #convertMessage(message: ChatOptions['messages'][number]): Promise<NonNullable<_OllamaChatConfig['messages']>[number]> {
    const { role, content } = message;
    const encodeContent = (content: string | ContentPart[]): string => {
      if (_.isString(content)) return content;
      return content.filter(c => c.type === 'text').map(c => 'text' in c ? c.text : '').join('\n');
    };
    const fetchImages = async (content: string | ContentPart[]) => {
      if (_.isString(content)) return undefined;
      const images = content.filter(c => c.type === 'image').map(async c => {
        const url = c.image.url;
        const matches = url.match(/^data:([^;]+);base64,(.+)$/);
        if (matches) {
          return matches[2];
        } else {
          const response = await fetch(url);
          if (!response.ok) {
            throw new Error(`Failed to fetch URL: ${response.statusText}`);
          }
          const arrayBuffer = await response.arrayBuffer();
          return Buffer.from(arrayBuffer).toString('base64');
        };
      });
      return _.isEmpty(images) ? undefined : Promise.all(images);
    };
    switch (role) {
      case 'user':
        return {
          role: 'user',
          content: encodeContent(content),
          images: await fetchImages(content),
        };
      case 'assistant':
        return {
          role: 'model',
          content,
          thinking: message.reasoning,
          tool_calls: message.tool_calls?.map(call => ({
            id: call.id,
            function: {
              name: call.name,
              arguments: JSON.parse(call.arguments),
            },
          })),
        };
      case 'tool':
        return {
          role: 'tool',
          content: encodeContent(content),
          images: await fetchImages(content),
          tool_name: message.name,
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
  }: OllamaChatConfig) {

    const response = await this.client.chat({
      ...options,
      stream: true,
      model,
      messages: _.compact([
        systemMessage && { role: 'system', content: systemMessage },
        ...await Promise.all(messages.map(msg => this.#convertMessage(msg))),
      ]),
      tools: tools ? _.map(tools, tool => ({
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
      })) : undefined,
    });

    const now = Date.now();
    const toolCallIds = new Map<number, string>();
    let counter = 0;
    let usage;

    for await (const { message: { content, thinking, tool_calls }, ...data } of response) {
      if (content) yield { type: 'content', content } as const;
      if (thinking) yield { type: 'reasoning', reasoning: thinking } as const;
      if (tool_calls) {
        const _tool_calls: (ToolCall & { id?: string; function: { index?: number; } })[] = tool_calls;
        for (const { id, function: call } of _tool_calls) {
          const index = call.index ?? counter++;
          if (!toolCallIds.has(index)) toolCallIds.set(index, id ?? `tool-${now}-${index}`);
          yield {
            type: 'tool_call',
            tool_call_id: toolCallIds.get(index)!,
            name: call.name,
            arguments: _.isString(call.arguments) ? call.arguments : JSON.stringify(call.arguments),
          } as const;
        }
      }
      if (!_.isNil(data.prompt_eval_count) || !_.isNil(data.eval_count)) {
        usage = {
          prompt_tokens: data.prompt_eval_count,
          completion_tokens: data.eval_count,
          total_tokens: (data.prompt_eval_count ?? 0) + (data.eval_count ?? 0),
        };
      }
    }

    if (usage) {
      yield { type: 'usage', usage } as const;
    }
  }
};