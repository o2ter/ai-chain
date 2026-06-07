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
import Anthropic, { ClientOptions } from '@anthropic-ai/sdk';
import { ClientProvider } from '../../client/provider';
import { ChatOptions, EmbedOptions } from '../../client/types';

type _AnthropicChatConfig = Parameters<Anthropic['messages']['create']>[0];
type AnthropicChatConfig = Omit<_AnthropicChatConfig, keyof ChatOptions | 'stream'> & ChatOptions;

export class AnthropicProvider extends ClientProvider {

  client: Anthropic;

  constructor(options: ClientOptions) {
    super();
    this.client = new Anthropic(options);
  };

  async* models() {
    for await (const model of this.client.models.list()) {
      yield { name: model.id };
    }
  }

  async embeddings({ }: EmbedOptions): Promise<any> {
    throw new Error('Anthropic API does not support embeddings');
  }

  #createChatParams({
    model,
    systemMessage,
    messages,
    tools,
    ...options
  }: AnthropicChatConfig): _AnthropicChatConfig {
    return {
      model,
      system: systemMessage,
      messages: messages.map(msg => {
        if (msg.role === 'user') {
          return { role: 'user', content: msg.content };
        } else if (msg.role === 'assistant') {
          return { role: 'assistant', content: msg.content };
        }
        return null;
      }),
      tools: tools ? _.map(tools, tool => ({
        name: tool.name,
        description: tool.description,
        input_schema: tool.parameters,
      })) : undefined,
      ...options
    };
  }

  async chat({ signal, ...options }: AnthropicChatConfig) {
    const response = await this.client.messages.create({
      ...this.#createChatParams(options),
      stream: false,
    }, { signal });

    const { content: _content, usage } = response;

    const content = _.filter(_content, x => x.type !== 'tool_use');
    const tool_calls = _.filter(_content, x => x.type === 'tool_use').map(x => ({
      id: x.id,
      name: x.name,
      arguments: x.input,
    }));

    return {
      content,
      tool_calls,
      usage: {
        completion_tokens: usage.input_tokens,
        prompt_tokens: usage.input_tokens,
        total_tokens: usage.input_tokens + usage.output_tokens,
      },
    };
  }

  async *chatStream({ signal, ...options }: AnthropicChatConfig) {
    const stream = await this.client.messages.create({
      ...this.#createChatParams(options),
      stream: true,
    }, { stream: true, signal });

    let usage;
    let currentToolId: string | undefined;

    for await (const part of stream) {
      switch (part.type) {
        case 'content_block_delta':
          switch (part.delta.type) {
            case 'thinking_delta':
              yield { type: 'reasoning', reasoning: part.delta.thinking } as const;
              break;
            case 'text_delta':
              yield { type: 'content', content: part.delta.text } as const;
              break;
            case 'input_json_delta':
              if (currentToolId) {
                yield {
                  type: 'tool_call',
                  tool_call_id: currentToolId,
                  arguments: part.delta.partial_json,
                } as const;
              }
              break;
            default:
              break;
          }
          break;
        case 'content_block_start':
          switch (part.content_block.type) {
            case 'tool_use':
              currentToolId = part.content_block.id;
              yield {
                type: 'tool_call',
                tool_call_id: currentToolId,
                name: part.content_block.name,
              } as const;
              break;
            default:
              break;
          }
          break;
        case 'message_start':
          usage = part.message.usage;
          break;
        case 'message_delta':
          usage = part.usage;
          break;
        default: break;
      }
    }

    if (usage) {
      yield {
        type: 'usage',
        usage: {
          completion_tokens: usage.output_tokens,
          prompt_tokens: usage.input_tokens ?? 0,
          total_tokens: (usage.input_tokens ?? 0) + usage.output_tokens,
        },
      } as const;
    }
  }
}