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

  #convertMessage(message: ChatOptions['messages'][number]): _AnthropicChatConfig['messages'][number] {
    const { role, content } = message;
    switch (role) {
      case 'user':
        return {
          role: 'user',
          content: _.isString(content)
            ? content
            : content.map(c => {
              switch (c.type) {
                case 'text':
                  return { type: 'text', text: c.text };
                case 'image_url':
                  const url = c.image_url.url;
                  const matches = url.match(/^data:([^;]+);base64,(.+)$/);
                  if (matches) {
                    return {
                      type: 'image',
                      source: {
                        type: 'base64',
                        media_type: matches[1] as any,
                        data: matches[2],
                      },
                    };
                  } else {
                    return {
                      type: 'image',
                      source: {
                        type: 'url',
                        url: c.image_url.url,
                      },
                    };
                  }
                default:
                  throw new Error(`Unsupported content type: ${(c as any).type}`);
              }
            }),
        };
      case 'assistant':
        return { role: 'assistant', content };
      case 'tool':
        return {
          role: 'user',
          content: [{
            type: 'tool_result',
            tool_use_id: message.tool_call_id,
            content,
          }],
        };
      default:
        throw new Error(`Unsupported message role: ${role}`);
    }
  }

  async *chat({
    model,
    systemMessage,
    messages,
    tools,
    signal,
    ...options
  }: AnthropicChatConfig) {
    const stream = await this.client.messages.create({
      ...options,
      stream: true,
      model,
      system: systemMessage,
      messages: messages.map(msg => this.#convertMessage(msg)),
      tools: tools ? _.map(tools, tool => ({
        name: tool.name,
        description: tool.description,
        input_schema: tool.parameters,
      })) : undefined,
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